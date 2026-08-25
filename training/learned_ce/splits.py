"""Template-aware train/validation/test splits."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import random


def assign_template_splits(
    templates: list[str],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Assign whole templates to train/val/test (never split rows within a template)."""
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    unique_templates = sorted(set(templates))
    rng = random.Random(seed)
    rng.shuffle(unique_templates)

    train_cut = int(len(unique_templates) * train_ratio)
    val_cut = train_cut + int(len(unique_templates) * val_ratio)

    split_map: dict[str, str] = {}
    for idx, template in enumerate(unique_templates):
        if idx < train_cut:
            split_map[template] = "train"
        elif idx < val_cut:
            split_map[template] = "val"
        else:
            split_map[template] = "test"
    return split_map


def stable_query_id(template: str, params: dict[str, object], idx: int) -> str:
    payload = f"{template}|{sorted(params.items())}|{idx}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{template}_{digest}"
