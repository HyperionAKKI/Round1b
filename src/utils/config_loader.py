"""
Config loader that normalises every supported Round-1B JSON variant
into a single canonical dictionary.

Expected variants
-----------------
1. Legacy (flat):
   {
     "persona": "…",
     "job_to_be_done": "…",
     "documents": ["a.pdf", "b.pdf"]
   }

2. New (nested – your screenshot):
   {
     "challenge_info": { … },
     "persona": { "role": "Travel Planner" },
     "job_to_be_done": { "task": "Plan a trip …" },
     "documents": [
        { "filename": "doc.pdf", "title": "Nice read" },
        …
     ]
   }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


class ConfigLoader:
    """Load & normalise either JSON schema into a flat internal dict."""

    REQUIRED_KEYS = {"persona", "job_to_be_done", "documents"}

    def __init__(self, json_path: str | Path) -> None:
        self.path = Path(json_path)

    # ------------------------------------------------------------------ #
    # public helpers
    # ------------------------------------------------------------------ #
    def load(self) -> Dict[str, Any]:
        raw = self._read_json(self.path)
        clean = self._normalise(raw)
        self._validate(clean)
        return clean

    # ------------------------------------------------------------------ #
    # private helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def _normalise(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a dict shaped as:
        {
           "persona": "<string>",
           "job_to_be_done": "<string>",
           "documents": ["file1.pdf", "file2.pdf"]
        }
        """
        # --------  persona  -------- #
        persona = raw.get("persona")
        if isinstance(persona, dict):
            persona = persona.get("role", "")
        persona = str(persona).strip()

        # --------  job/task  -------- #
        job = raw.get("job_to_be_done") or raw.get("job")
        if isinstance(job, dict):
            # common keys in various datasets
            job = job.get("task") or job.get("description") or job.get("title") or ""
        job = str(job).strip()

        # --------  documents  -------- #
        docs: List[str] = []
        for item in raw.get("documents", []):
            if isinstance(item, str):
                docs.append(item)
            elif isinstance(item, dict):
                filename = item.get("filename") or item.get("file") or item.get("name")
                if filename:
                    docs.append(str(filename).strip())
        docs = [d for d in docs if d]  # drop empties / None

        return {"persona": persona, "job_to_be_done": job, "documents": docs}

    def _validate(self, cfg: Dict[str, Any]) -> None:
        missing = [k for k in self.REQUIRED_KEYS if not cfg.get(k)]
        if missing:
            raise ValueError(f"Missing required config fields: {', '.join(missing)}")
