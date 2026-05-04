import os
import sys
from collections.abc import Iterable
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def progress(iterable: Iterable[T], **kwargs) -> Iterable[T]:
    """Return a tqdm-wrapped iterable only when progress output is safe."""
    if _env_flag("LTX_DISABLE_TQDM"):
        return iterable
    if not _env_flag("LTX_FORCE_TQDM"):
        stream = kwargs.get("file", sys.stderr)
        isatty = getattr(stream, "isatty", None)
        if not callable(isatty) or not isatty():
            return iterable

    try:
        return tqdm(iterable, **kwargs)
    except (OSError, ValueError):
        return iterable
