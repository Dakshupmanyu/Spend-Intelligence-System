"""Blocked fuzzy matching must agree with brute force at production thresholds and stay fast."""
from __future__ import annotations

import random
import string
import time

import pandas as pd

from spend_engine.identity import BLOCK_CAP, _mutual_fuzzy_clusters, assign_entity_keys
from spend_engine.normalize import similarity_ratio


def _brute_force(names: list[str], threshold: float) -> dict[str, str]:
    """The original all-pairs algorithm, kept here as the reference."""
    unique = sorted(set(names))
    nearest = {}
    for name in unique:
        ranked = sorted(((similarity_ratio(name, o), o) for o in unique if o != name), reverse=True)
        if not ranked or ranked[0][0] < threshold or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 3):
            nearest[name] = None
        else:
            nearest[name] = (ranked[0][1], ranked[0][0])
    parent = {n: n for n in unique}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for name, choice in nearest.items():
        if choice is None:
            continue
        other, score = choice
        back = nearest.get(other)
        if back is None or back[0] != name or abs(back[1] - score) > 1e-9:
            continue
        a, b = find(name), find(other)
        parent[a] = parent[b] = min(a, b)
    return {n: find(n) for n in unique}


def _names(seed: int = 3) -> list[str]:
    rng = random.Random(seed)
    base = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(8, 14))) + " " + rng.choice(["mining", "pty", "supplies"])
            for _ in range(250)]
    names = list(base)
    for b in base[:120]:
        i = rng.randrange(len(b))
        names.append(b[:i] + rng.choice(string.ascii_lowercase) + b[i + 1:])  # one-character typo
    return names


def test_blocked_clusters_match_brute_force_at_production_threshold():
    names = _names()
    assert _mutual_fuzzy_clusters(names, 97.0) == _brute_force(names, 97.0)


def test_generic_oversized_block_does_not_crash_or_blow_up():
    # every name shares the token "valve": one block far larger than BLOCK_CAP
    names = [f"valve {''.join(random.Random(i).choices(string.ascii_lowercase, k=8))}" for i in range(BLOCK_CAP * 3)]
    names += [names[0] + "x"]  # near duplicate of the first name
    clusters = _mutual_fuzzy_clusters(names, 90.0)
    assert len(clusters) == len(set(names))


def test_idless_dedupe_scales_roughly_linearly():
    rng = random.Random(1)
    n = 4000
    frame = pd.DataFrame({
        "record_id": [f"R{i}" for i in range(n)], "supplier_id": "", "item_id": "",
        "supplier_name": ["".join(rng.choices(string.ascii_lowercase, k=9)) + " mining" for _ in range(n)],
        "description": [" ".join(rng.choices(["valve", "pump", "seal", "belt"], k=3)) + f" {rng.randint(1000, 9999)}" for _ in range(n)],
        "source": "C", "source_namespace": "C", "item_namespace": "C",
    })
    started = time.perf_counter()
    out = assign_entity_keys(frame)
    elapsed = time.perf_counter() - started
    assert elapsed < 20  # the all-pairs version needed ~150 s for this size
    assert len(out) == n


def test_known_typo_variants_still_merge():
    frame = pd.DataFrame({
        "record_id": ["1", "2", "3"], "supplier_id": "", "item_id": "",
        "supplier_name": ["Zanmore Hydraulic Mining (Pty) Ltd", "ZANMORE HYDRAULIC MINING", "Zanmore Hydraulic Minin"],
        "description": ["", "", ""], "source": "C", "source_namespace": "C", "item_namespace": "C",
    })
    out = assign_entity_keys(frame)
    assert out["supplier_key"].nunique() <= 2 and out.loc[0, "supplier_key"] == out.loc[1, "supplier_key"]
