from veripet.research.pair_mining import (
    assert_no_identity_leakage,
    build_balanced_pairs_for_split_rows,
    build_pairs_for_split_rows,
    build_verification_pairs,
)


def _rows():
    return [
        {"filename": "a1.jpg", "label": "a", "breed": "Siamese"},
        {"filename": "a2.jpg", "label": "a", "breed": "Siamese"},
        {"filename": "b1.jpg", "label": "b", "breed": "Siamese"},
        {"filename": "b2.jpg", "label": "b", "breed": "Siamese"},
        {"filename": "c1.jpg", "label": "c", "breed": "Bengal"},
        {"filename": "c2.jpg", "label": "c", "breed": "Bengal"},
        {"filename": "d1.jpg", "label": "d", "breed": "breed_unknown"},
        {"filename": "d2.jpg", "label": "d", "breed": "breed_unknown"},
        {"filename": "e1.jpg", "label": "e", "breed": "Persian"},
        {"filename": "e2.jpg", "label": "e", "breed": "Persian"},
    ]


def test_pair_builder_generates_required_pair_types_without_leakage():
    result = build_verification_pairs(
        _rows(),
        val_fraction=0.4,
        test_fraction=0.4,
        negatives_per_positive=2,
        seed=7,
    )

    assert_no_identity_leakage(
        {
            "train": result.train_ids,
            "verification_val": result.val_ids,
            "verification_test": result.test_ids,
        }
    )
    pair_types = {pair.pair_type for pair in result.val_pairs + result.test_pairs}
    assert "positive" in pair_types
    assert "negative_same_breed" in pair_types
    assert "negative_diff_breed" in pair_types or "breed_unknown" in pair_types


def test_identity_leakage_detection_raises():
    try:
        assert_no_identity_leakage({"train": ["a"], "verification_val": ["a"]})
    except ValueError as exc:
        assert "Identity leakage" in str(exc)
    else:
        raise AssertionError("Expected leakage detection to raise")


def test_build_pairs_for_fixed_split_rows():
    pairs = build_pairs_for_split_rows(_rows()[:6], negatives_per_positive=2, seed=7)
    pair_types = {pair.pair_type for pair in pairs}
    assert "positive" in pair_types
    assert "negative_same_breed" in pair_types
    assert "negative_diff_breed" in pair_types


def test_build_balanced_pairs_for_split_rows_hits_requested_mix():
    rows = [
        {"filename": "a1.jpg", "label": "a", "breed": "Siamese"},
        {"filename": "a2.jpg", "label": "a", "breed": "Siamese"},
        {"filename": "b1.jpg", "label": "b", "breed": "Siamese"},
        {"filename": "b2.jpg", "label": "b", "breed": "Siamese"},
        {"filename": "c1.jpg", "label": "c", "breed": "Bengal"},
        {"filename": "c2.jpg", "label": "c", "breed": "Bengal"},
        {"filename": "d1.jpg", "label": "d", "breed": "Persian"},
        {"filename": "d2.jpg", "label": "d", "breed": "Persian"},
    ]
    pairs = build_balanced_pairs_for_split_rows(rows, seed=7)
    counts = {}
    for pair in pairs:
        counts[pair.pair_type] = counts.get(pair.pair_type, 0) + 1
    assert counts["positive"] == 2
    assert counts["negative_same_breed"] == 1
    assert counts["negative_diff_breed"] == 1
