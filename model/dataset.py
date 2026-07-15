"""Load cached Poker44 benchmark releases into (group, label, date, split) examples.

Each example is one "chunk group": a list of 30-40 hand dicts, all from the same
hero (one player/session), with a single ground-truth label (1=bot, 0=human).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


@dataclass
class Example:
    hands: List[Dict[str, Any]]   # the chunk group (miner-visible hands)
    label: int                    # 1 = bot, 0 = human
    source_date: str
    split: str                    # "train" | "validation" (as released)
    chunk_id: str
    group_index: int


def load_examples(data_dir: str = DATA_DIR) -> List[Example]:
    examples: List[Example] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path) as fh:
            doc = json.load(fh)
        if not doc.get("success", True) and "data" not in doc:
            continue
        data = doc["data"]
        source_date = data["sourceDate"]
        for record in data["chunks"]:
            split = record.get("split", "train")
            chunk_id = record.get("chunkId", "")
            groups = record["chunks"]               # list of groups (each a list of hands)
            labels = record["groundTruth"]           # one label per group, aligned by index
            for gi, (group, label) in enumerate(zip(groups, labels)):
                examples.append(
                    Example(
                        hands=group,
                        label=int(label),
                        source_date=source_date,
                        split=split,
                        chunk_id=chunk_id,
                        group_index=gi,
                    )
                )
    return examples


if __name__ == "__main__":
    from collections import Counter

    ex = load_examples()
    print(f"total examples: {len(ex)}")
    print("label balance:", Counter(e.label for e in ex))
    print("split balance:", Counter(e.split for e in ex))
    print("dates:", len(set(e.source_date for e in ex)))
    sizes = [len(e.hands) for e in ex]
    print(f"group sizes: min={min(sizes)} max={max(sizes)} mean={sum(sizes)/len(sizes):.1f}")
    # sanity: per-date balance
    by_date_label = Counter((e.source_date, e.label) for e in ex)
    bad = [d for d in set(e.source_date for e in ex)
           if by_date_label[(d, 1)] == 0 or by_date_label[(d, 0)] == 0]
    print("dates missing a class:", bad)
    total_hands = sum(sizes)
    print(f"total hands across all groups: {total_hands}")
