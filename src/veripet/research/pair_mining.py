"""Pair generation utilities for verification experiments."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _group_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def _pair_type(breed1: str, breed2: str) -> str:
    if breed1 in {"breed_unknown", "Unknown", ""} or breed2 in {"breed_unknown", "Unknown", ""}:
        return "breed_unknown"
    if breed1 == breed2:
        return "negative_same_breed"
    return "negative_diff_breed"


def _is_excluded_breed(breed: str, excluded_breeds: Set[str]) -> bool:
    return breed in excluded_breeds


def _has_same_breed(ids: Sequence[str], identity_meta: Dict[str, str]) -> bool:
    counts: Dict[str, int] = {}
    for identity in ids:
        breed = identity_meta[identity]
        if breed in {"breed_unknown", "Unknown", ""}:
            continue
        counts[breed] = counts.get(breed, 0) + 1
    return any(count >= 2 for count in counts.values())


def _has_diff_breed(ids: Sequence[str], identity_meta: Dict[str, str]) -> bool:
    breeds = {
        identity_meta[identity]
        for identity in ids
        if identity_meta[identity] not in {"breed_unknown", "Unknown", ""}
    }
    return len(breeds) >= 2


def _enrich_eval_splits(
    val_ids: List[str], test_ids: List[str], train_ids: List[str], identity_meta: Dict[str, str]
) -> None:
    for split_ids in (val_ids, test_ids):
        if not split_ids:
            continue
        if not _has_same_breed(split_ids, identity_meta):
            split_breeds = {identity_meta[identity] for identity in split_ids}
            for candidate in list(train_ids):
                candidate_breed = identity_meta[candidate]
                if candidate_breed in split_breeds and candidate_breed not in {"breed_unknown", "Unknown", ""}:
                    split_ids.append(candidate)
                    train_ids.remove(candidate)
                    break
        if not _has_diff_breed(split_ids, identity_meta):
            split_breeds = {identity_meta[identity] for identity in split_ids if identity_meta[identity] != "breed_unknown"}
            for candidate in list(train_ids):
                candidate_breed = identity_meta[candidate]
                if candidate_breed not in {"breed_unknown", "Unknown", ""} and candidate_breed not in split_breeds:
                    split_ids.append(candidate)
                    train_ids.remove(candidate)
                    break


@dataclass(frozen=True)
class PairRecord:
    filename1: str
    filename2: str
    label: int
    id1: str
    id2: str
    breed1: str
    breed2: str
    pair_type: str


@dataclass(frozen=True)
class PairSplitResult:
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    val_pairs: List[PairRecord]
    test_pairs: List[PairRecord]
    stats: Dict[str, object]


def assert_no_identity_leakage(split_to_ids: Dict[str, Sequence[str]]) -> None:
    split_names = list(split_to_ids)
    for idx, first in enumerate(split_names):
        first_ids = set(split_to_ids[first])
        for second in split_names[idx + 1 :]:
            overlap = first_ids & set(split_to_ids[second])
            if overlap:
                raise ValueError(f"Identity leakage between {first} and {second}: {sorted(overlap)}")


def build_verification_pairs(
    rows: Sequence[Dict[str, str]],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    negatives_per_positive: int = 2,
    seed: int = 42,
) -> PairSplitResult:
    by_identity = _group_by(rows, "label")
    identity_meta = {
        identity: identity_rows[0].get("breed", "breed_unknown")
        for identity, identity_rows in by_identity.items()
    }
    rng = random.Random(seed)

    total = len(by_identity)
    val_count = max(1, int(total * val_fraction))
    test_count = max(1, int(total * test_fraction))
    breed_groups = _group_by(
        [{"label": identity, "breed": breed} for identity, breed in identity_meta.items()],
        "breed",
    )
    grouped_identities = [
        sorted(group["label"] for group in breed_rows) for breed_rows in breed_groups.values()
    ]
    grouped_identities.sort(key=len, reverse=True)
    rng.shuffle(grouped_identities)

    val_ids: List[str] = []
    test_ids: List[str] = []
    train_ids: List[str] = []

    for group in grouped_identities:
        if len(val_ids) + len(group) <= val_count:
            val_ids.extend(group)
        elif len(test_ids) + len(group) <= test_count:
            test_ids.extend(group)
        else:
            train_ids.extend(group)

    leftovers = [
        identity
        for identity in by_identity
        if identity not in set(val_ids) | set(test_ids) | set(train_ids)
    ]
    for identity in leftovers:
        if len(val_ids) < val_count:
            val_ids.append(identity)
        elif len(test_ids) < test_count:
            test_ids.append(identity)
        else:
            train_ids.append(identity)

    _enrich_eval_splits(val_ids, test_ids, train_ids, identity_meta)

    split_ids = {"train": train_ids, "verification_val": val_ids, "verification_test": test_ids}
    assert_no_identity_leakage(split_ids)

    val_pairs = build_pairs_for_split_rows(
        [row for identity in val_ids for row in by_identity[identity]],
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )
    test_pairs = build_pairs_for_split_rows(
        [row for identity in test_ids for row in by_identity[identity]],
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )
    stats = {
        "seed": seed,
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "pair_counts": {"verification_val": len(val_pairs), "verification_test": len(test_pairs)},
        "pair_type_counts": {
            "verification_val": _count_pair_types(val_pairs),
            "verification_test": _count_pair_types(test_pairs),
        },
    }
    return PairSplitResult(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        stats=stats,
    )


def _count_pair_types(pairs: Sequence[PairRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for pair in pairs:
        counts[pair.pair_type] = counts.get(pair.pair_type, 0) + 1
    return counts


def pair_records_to_rows(pairs: Sequence[PairRecord]) -> List[Dict[str, object]]:
    return [asdict(pair) for pair in pairs]


def build_pairs_for_split_rows(
    rows: Sequence[Dict[str, str]],
    negatives_per_positive: int = 2,
    seed: int = 42,
) -> List[PairRecord]:
    rng = random.Random(seed)
    by_local_identity = _group_by(rows, "label")
    pairs: List[PairRecord] = []

    for identity, identity_rows in by_local_identity.items():
        for left, right in itertools.combinations(identity_rows, 2):
            pairs.append(
                PairRecord(
                    filename1=left["filename"],
                    filename2=right["filename"],
                    label=1,
                    id1=identity,
                    id2=identity,
                    breed1=left.get("breed", "breed_unknown"),
                    breed2=right.get("breed", "breed_unknown"),
                    pair_type="positive",
                )
            )

    identity_pool = list(by_local_identity.keys())
    for identity in identity_pool:
        anchor = by_local_identity[identity][0]
        same_breed: List[str] = []
        diff_breed: List[str] = []
        unknown_breed: List[str] = []
        for candidate in identity_pool:
            if candidate == identity:
                continue
            candidate_breed = by_local_identity[candidate][0].get("breed", "breed_unknown")
            anchor_breed = anchor.get("breed", "breed_unknown")
            if "breed_unknown" in (anchor_breed, candidate_breed):
                unknown_breed.append(candidate)
            elif candidate_breed == anchor_breed:
                same_breed.append(candidate)
            else:
                diff_breed.append(candidate)

        rng.shuffle(same_breed)
        rng.shuffle(diff_breed)
        rng.shuffle(unknown_breed)
        ordered_candidates = same_breed + diff_breed + unknown_breed

        for negative_identity in ordered_candidates[:negatives_per_positive]:
            other = by_local_identity[negative_identity][0]
            pairs.append(
                PairRecord(
                    filename1=anchor["filename"],
                    filename2=other["filename"],
                    label=0,
                    id1=identity,
                    id2=negative_identity,
                    breed1=anchor.get("breed", "breed_unknown"),
                    breed2=other.get("breed", "breed_unknown"),
                    pair_type=_pair_type(
                        anchor.get("breed", "breed_unknown"),
                        other.get("breed", "breed_unknown"),
                    ),
                )
            )
    return pairs


def build_balanced_pairs_for_split_rows(
    rows: Sequence[Dict[str, str]],
    seed: int = 42,
    positive_fraction: float = 0.50,
    same_breed_fraction: float = 0.25,
    diff_breed_fraction: float = 0.25,
    excluded_breeds: Optional[Set[str]] = None,
) -> List[PairRecord]:
    if round(positive_fraction + same_breed_fraction + diff_breed_fraction, 6) != 1.0:
        raise ValueError("Pair fractions must sum to 1.0")

    excluded_breeds = excluded_breeds or {"Unknown", "breed_unknown", ""}
    rng = random.Random(seed)
    by_identity = _group_by(rows, "label")
    identity_breed = {
        identity: identity_rows[0].get("breed", "breed_unknown")
        for identity, identity_rows in by_identity.items()
    }
    eligible_identities = [
        identity
        for identity, breed in identity_breed.items()
        if not _is_excluded_breed(breed, excluded_breeds)
    ]
    if len(eligible_identities) < 2:
        return []

    positive_candidates: List[PairRecord] = []
    by_breed: Dict[str, List[str]] = {}
    for identity in eligible_identities:
        breed = identity_breed[identity]
        by_breed.setdefault(breed, []).append(identity)
        identity_rows = by_identity[identity]
        for left, right in itertools.combinations(identity_rows, 2):
            positive_candidates.append(
                PairRecord(
                    filename1=left["filename"],
                    filename2=right["filename"],
                    label=1,
                    id1=identity,
                    id2=identity,
                    breed1=breed,
                    breed2=breed,
                    pair_type="positive",
                )
            )

    available_positive = len(positive_candidates)
    same_identity_pairs = sum(comb(len(ids), 2) for ids in by_breed.values() if len(ids) >= 2)
    total_known_identity_pairs = comb(len(eligible_identities), 2)
    diff_identity_pairs = max(0, total_known_identity_pairs - same_identity_pairs)

    if available_positive == 0 or same_identity_pairs == 0 or diff_identity_pairs == 0:
        return []

    positive_target = min(
        available_positive,
        int((same_identity_pairs * positive_fraction) / same_breed_fraction),
        int((diff_identity_pairs * positive_fraction) / diff_breed_fraction),
    )
    same_target = min(same_identity_pairs, int((positive_target * same_breed_fraction) / positive_fraction))
    diff_target = min(diff_identity_pairs, int((positive_target * diff_breed_fraction) / positive_fraction))
    positive_target = min(
        available_positive,
        int((same_target * positive_fraction) / same_breed_fraction),
        int((diff_target * positive_fraction) / diff_breed_fraction),
    )
    if min(positive_target, same_target, diff_target) <= 0:
        return []

    rng.shuffle(positive_candidates)
    positive_pairs = positive_candidates[:positive_target]
    same_pairs = _sample_negative_pairs(
        by_identity=by_identity,
        identity_breed=identity_breed,
        target_count=same_target,
        rng=rng,
        same_breed=True,
        excluded_breeds=excluded_breeds,
    )
    diff_pairs = _sample_negative_pairs(
        by_identity=by_identity,
        identity_breed=identity_breed,
        target_count=diff_target,
        rng=rng,
        same_breed=False,
        excluded_breeds=excluded_breeds,
    )
    return positive_pairs + same_pairs + diff_pairs


def _sample_negative_pairs(
    by_identity: Dict[str, List[Dict[str, str]]],
    identity_breed: Dict[str, str],
    target_count: int,
    rng: random.Random,
    same_breed: bool,
    excluded_breeds: Set[str],
) -> List[PairRecord]:
    pairs: List[PairRecord] = []
    seen_identity_pairs: Set[Tuple[str, str]] = set()

    if same_breed:
        breed_groups: Dict[str, List[str]] = {}
        for identity, breed in identity_breed.items():
            if _is_excluded_breed(breed, excluded_breeds):
                continue
            breed_groups.setdefault(breed, []).append(identity)
        weighted_breeds: List[str] = []
        for breed, ids in breed_groups.items():
            if len(ids) >= 2:
                weighted_breeds.extend([breed] * max(1, len(ids) // 2))
        attempts = 0
        max_attempts = max(5000, target_count * 20)
        while len(pairs) < target_count and attempts < max_attempts and weighted_breeds:
            attempts += 1
            breed = rng.choice(weighted_breeds)
            left_id, right_id = sorted(rng.sample(breed_groups[breed], 2))
            key = (left_id, right_id)
            if key in seen_identity_pairs:
                continue
            seen_identity_pairs.add(key)
            left = rng.choice(by_identity[left_id])
            right = rng.choice(by_identity[right_id])
            pairs.append(
                PairRecord(
                    filename1=left["filename"],
                    filename2=right["filename"],
                    label=0,
                    id1=left_id,
                    id2=right_id,
                    breed1=breed,
                    breed2=breed,
                    pair_type="negative_same_breed",
                )
            )
        return pairs

    eligible = [
        identity
        for identity, breed in identity_breed.items()
        if not _is_excluded_breed(breed, excluded_breeds)
    ]
    attempts = 0
    max_attempts = max(10000, target_count * 30)
    while len(pairs) < target_count and attempts < max_attempts and len(eligible) >= 2:
        attempts += 1
        left_id, right_id = rng.sample(eligible, 2)
        if identity_breed[left_id] == identity_breed[right_id]:
            continue
        key = tuple(sorted((left_id, right_id)))
        if key in seen_identity_pairs:
            continue
        seen_identity_pairs.add(key)
        left = rng.choice(by_identity[left_id])
        right = rng.choice(by_identity[right_id])
        pairs.append(
            PairRecord(
                filename1=left["filename"],
                filename2=right["filename"],
                label=0,
                id1=left_id,
                id2=right_id,
                breed1=identity_breed[left_id],
                breed2=identity_breed[right_id],
                pair_type="negative_diff_breed",
            )
        )
    return pairs


def save_pair_stats(stats: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True), encoding="utf-8")
