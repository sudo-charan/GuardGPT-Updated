import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

try:
    import faiss
except ImportError:  # JSONL mode can use NumPy search without FAISS.
    faiss = None
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_JSONL_DATASET = PROJECT_ROOT / "guardgpt_dataset.jsonl"
DATASET_PATH = Path(
    os.getenv(
        "GUARDGPT_DATASET",
        str(_JSONL_DATASET if _JSONL_DATASET.is_file() else PROJECT_ROOT / "data" / "guardgpt_augmented_clean.json"),
    )
)
CACHE_DIR = PROJECT_ROOT / "cache"
INDEX_PATH = PROJECT_ROOT / "data" / "guardgpt_faiss.index"
RECORDS_PATH = PROJECT_ROOT / "data" / "guardgpt_id_map.json"
MODEL_NAME = "all-MiniLM-L6-v2"


class DatasetLoader:
    """Loads dataset and executes vector search via Sentence-BERT + FAISS."""

    def __init__(
        self,
        path: str | Path = DATASET_PATH,
        model_name: str = MODEL_NAME,
        max_records: Optional[int] = None,  # Full dataset indexing enabled
    ) -> None:
        self._path = Path(path)
        self._model_name = model_name
        self.max_records = max_records
        self._records: list[dict] = []
        self._texts: list[str] = []
        self._model: Optional[SentenceTransformer] = None
        self._index = None
        self._embeddings = None
        self._jsonl_mode = self._path.suffix.lower() == ".jsonl"
        self._loaded = False
        self._load_time = 0.0

    def load(self) -> None:
        if self._loaded:
            return

        if self._jsonl_mode:
            self._load_jsonl()
        else:
            self._load_prebuilt()

    def _load_jsonl(self) -> None:
        """Load the supplied JSONL dataset and build an in-memory vector index."""
        start = time.monotonic()
        if not self._path.is_file():
            raise FileNotFoundError(f"Required dataset artifact missing: {self._path}")
        records = []
        with self._path.open(encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at line {line_number}") from error
                if not isinstance(raw, dict):
                    raise ValueError(f"JSONL line {line_number} must be an object.")
                text = raw.get("input_text") or raw.get("prompt") or raw.get("text") or raw.get("request")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"JSONL line {line_number} has no prompt text.")
                record = dict(raw)
                record.setdefault("request_id", f"dataset_{line_number}")
                record["input_text"] = text.strip()
                record.setdefault("intent", "safe" if str(record.get("target_verdict", "ALLOW")).upper() == "ALLOW" else "unsafe")
                record.setdefault("target_verdict", "ALLOW" if record["intent"] == "safe" else "BLOCK")
                scores = record.get("category_scores")
                record["category_scores"] = dict(scores) if isinstance(scores, dict) else {
                    "harm": 0.0, "toxicity": 0.0, "jailbreak": 0.0, "prompt_injection": 0.0
                }
                records.append(record)
        if not records:
            raise ValueError("JSONL dataset is empty.")
        if self.max_records is not None:
            records = records[: self.max_records]
        try:
            model = SentenceTransformer(self._model_name, local_files_only=True)
        except Exception:
            model = SentenceTransformer(self._model_name)
        embeddings = model.encode(
            [record["input_text"] for record in records],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self._model, self._records, self._texts = model, records, [r["input_text"] for r in records]
        self._embeddings = embeddings
        self._index = faiss.IndexFlatIP(embeddings.shape[1]) if faiss is not None else None
        if self._index is not None:
            self._index.add(embeddings)
        self._load_time = time.monotonic() - start
        self._loaded = True
        logger.info("Loaded JSONL dataset: %d records", len(records))

    def _load_prebuilt(self) -> None:
        """Validate the three artifacts before publishing any loaded state."""
        start = time.monotonic()
        index_path = self._path.parent / "guardgpt_faiss.index"
        map_path = self._path.parent / "guardgpt_id_map.json"
        for path in (self._path, index_path, map_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required dataset artifact missing: {path}")
        if self.max_records is not None:
            raise ValueError("max_records is unsupported for a prebuilt index.")
        if self._model_name not in {MODEL_NAME, f"sentence-transformers/{MODEL_NAME}"}:
            raise ValueError(f"This index requires {MODEL_NAME}.")
        with self._path.open(encoding="utf-8-sig") as file:
            dataset = json.load(file)
        with map_path.open(encoding="utf-8-sig") as file:
            mapping = json.load(file)
        if not isinstance(dataset, list) or not dataset:
            raise ValueError("Dataset must be a nonempty JSON list.")
        if not isinstance(mapping, dict):
            raise ValueError("ID map must be a dictionary keyed by vector number.")
        index = faiss.read_index(str(index_path))
        if faiss is None:
            raise RuntimeError("FAISS is required for prebuilt dataset artifacts.")
        if not isinstance(index, faiss.IndexFlatIP) or index.d != 384:
            raise ValueError("Expected a 384-dimensional IndexFlatIP index.")
        if index.ntotal != len(dataset) or set(mapping) != {str(i) for i in range(index.ntotal)}:
            raise ValueError("Dataset, index and ID-map counts/keys do not match.")
        by_id = {}
        for record in dataset:
            if not isinstance(record, dict):
                raise ValueError("Each dataset record must be an object.")
            rid = record.get("request_id")
            if not isinstance(rid, str) or not rid or rid in by_id:
                raise ValueError("request_id must be a unique nonempty string.")
            if not isinstance(record.get("input_text"), str) or not record["input_text"].strip():
                raise ValueError(f"Missing input_text: {rid}")
            if record.get("intent") not in {"safe", "unsafe"}:
                raise ValueError(f"Expected safe/unsafe intent: {rid}")
            if record.get("target_verdict") not in {"ALLOW", "BLOCK", "SANITIZE"}:
                raise ValueError(f"Invalid target_verdict: {rid}")
            if (record["intent"] == "safe") != (record["target_verdict"] == "ALLOW"):
                raise ValueError(f"Conflicting intent and target_verdict: {rid}")
            scores = record.get("category_scores")
            if not isinstance(scores, dict) or not {"harm", "toxicity", "jailbreak", "prompt_injection"} <= scores.keys():
                raise ValueError(f"Missing category scores: {rid}")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
                raise ValueError(f"Invalid category score: {rid}")
            if record["intent"] == "safe" and any(scores[k] >= 0.5 for k in ("harm", "toxicity", "jailbreak", "prompt_injection")):
                raise ValueError(f"Safe label conflicts with high harm scores: {rid}")
            by_id[rid] = record
        records = [mapping[str(i)] for i in range(index.ntotal)]
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each ID-map value must be a full record.")
            rid = record.get("request_id")
            if not isinstance(rid, str) or rid in seen or by_id.get(rid) != record:
                raise ValueError("ID map has duplicate, missing or changed dataset records.")
            seen.add(rid)
        for i in np.linspace(0, index.ntotal - 1, min(100, index.ntotal), dtype=int):
            if not np.isclose(np.linalg.norm(index.reconstruct(int(i))), 1.0, atol=1e-3):
                raise ValueError("Index vectors must be L2 normalized.")
        try:
            model = SentenceTransformer(self._model_name, local_files_only=True)
        except Exception:
            model = SentenceTransformer(self._model_name)
        dimension_method = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        if dimension_method() != index.d:
            raise ValueError("Embedding model dimension does not match index.")
        self._model, self._index, self._records = model, index, records
        self._texts = [r["input_text"] for r in records]
        self._load_time = time.monotonic() - start
        self._loaded = True
        logger.info("Loaded validated prebuilt index: %d records", len(records))


    def query(self, text: str, top_k: int = 1) -> Optional[dict]:
        if not self._loaded:
            self.load()
        if not text or not text.strip():
            return None

        query_embedding = self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")

        top_k = max(1, min(top_k, len(self._records)))
        if self._index is not None:
            similarities, indices = self._index.search(query_embedding, top_k)
        else:
            scores = np.dot(self._embeddings, query_embedding[0])
            indices = np.argsort(scores)[::-1][:top_k][None, :]
            similarities = scores[indices]

        if indices.size == 0 or indices[0][0] < 0:
            return None

        best_index = int(indices[0][0])
        best_score = float(similarities[0][0])

        record = dict(self._records[best_index])
        record["_similarity"] = round(best_score, 4)
        return record

    def query_top_k(self, text: str, top_k: int = 5) -> list[dict]:
        if not self._loaded:
            self.load()
        if not text or not text.strip():
            return []

        query_embedding = self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")
        top_k = max(1, min(top_k, len(self._records)))
        if self._index is not None:
            similarities, indices = self._index.search(query_embedding, top_k)
        else:
            scores = np.dot(self._embeddings, query_embedding[0])
            indices = np.argsort(scores)[::-1][:top_k][None, :]
            similarities = scores[indices]

        results = []
        for score, index in zip(similarities[0], indices[0]):
            if index < 0:
                continue
            record = dict(self._records[int(index)])
            record["_similarity"] = round(float(score), 4)
            results.append(record)

        return results

    def clear_cache(self) -> None:
        """Release memory; never delete the supplied dataset artifacts."""
        self._index = None
        self._embeddings = None
        self._records = []
        self._texts = []
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def load_time(self) -> float:
        return self._load_time

    @property
    def embedding_dimension(self) -> int:
        if self._index is not None:
            return self._index.d
        return 0 if self._embeddings is None else int(self._embeddings.shape[1])
