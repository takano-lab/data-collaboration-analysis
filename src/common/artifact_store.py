from __future__ import annotations

import pickle
import hashlib
from pathlib import Path
from typing import Optional

from src.paths import OUTPUT_DIR


class ArtifactStore:
    """
    Thin wrapper around pickle-based persistence for Dataset/Intermediate/Integrated artifacts.
    """

    def __init__(self, *, base_dir: Optional[Path] = None, logger=None) -> None:
        self.base_dir = Path(base_dir or (OUTPUT_DIR / "artifacts"))
        self.logger = logger

    # ------------------------------------------------------------------ #
    def load(self, category: str, name: Optional[str]) -> object | None:
        path = self._path(category, name)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # pragma: no cover - defensive logging
            if self.logger:
                self.logger.warning("Failed to load artifact %s: %s", path, exc)
            return None

    def save(self, category: str, name: Optional[str], artifact: object) -> None:
        if artifact is None:
            return
        path = self._path(category, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(artifact, fh)
        if self.logger:
            self.logger.info("Saved %s artifact to %s", category, path)

    def prune(self, category: str, *, keep: int = 30) -> None:
        """
        Keep only the latest `keep` artifacts within a category directory, dropping
        older files based on modification time.
        """
        if keep <= 0:
            keep = 0
        directory = self.base_dir / self._safe_name(category)
        if not directory.exists():
            return
        try:
            candidates = [p for p in directory.iterdir() if p.is_file()]
        except FileNotFoundError:
            return
        if len(candidates) <= keep:
            return
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for old_path in candidates[keep:]:
            try:
                old_path.unlink()
                if self.logger:
                    self.logger.info("Pruned %s artifact: %s", category, old_path.name)
            except FileNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                if self.logger:
                    self.logger.warning("Failed to prune %s artifact %s: %s", category, old_path, exc)

    # ------------------------------------------------------------------ #
    def _path(self, category: str, name: Optional[str]) -> Path:
        safe_category = self._safe_name(category)
        safe_name = self._safe_name(name or category)
        return self.base_dir / safe_category / f"{safe_name}.pkl"

    @staticmethod
    def _safe_name(raw: Optional[str]) -> str:
        if not raw:
            return "default"
        text = str(raw)
        cleaned = ["_" if not (c.isalnum() or c in "-_") else c for c in text]
        slug = "".join(cleaned).strip("_")
        if not slug:
            slug = "default"
        max_len = 120
        if len(slug) > max_len:
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
            slug = f"{slug[:max_len-11]}_{digest}"
        return slug
