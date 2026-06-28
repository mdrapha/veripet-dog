from veripet.research.sampling import (
    SampleProfile,
    build_experiment_subset,
    build_identity_sample,
    build_local_sample_bundle,
)


def test_sampling_is_identity_preserving_and_reproducible():
    rows = [
        {"filename": f"a_{idx}.jpg", "label": "a", "breed": "Siamese"}
        for idx in range(3)
    ] + [
        {"filename": f"b_{idx}.jpg", "label": "b", "breed": "Bengal"}
        for idx in range(3)
    ] + [
        {"filename": f"c_{idx}.jpg", "label": "c", "breed": "Persian"}
        for idx in range(1)
    ]
    profile = SampleProfile(name="debug_sample", max_identities=2, min_images_per_identity=2)
    sample_one = build_identity_sample(rows, profile)
    sample_two = build_identity_sample(rows, profile)

    assert sample_one == sample_two
    assert {row["label"] for row in sample_one} <= {"a", "b"}
    assert all(sum(1 for row in sample_one if row["label"] == label) >= 2 for label in {"a", "b"})


def test_split_aware_sampling_and_experiment_fraction_work_by_identity():
    rows = [
        {"filename": f"train_a_{idx}.jpg", "label": "a", "breed": "Siamese", "split": "train"}
        for idx in range(3)
    ] + [
        {"filename": f"train_b_{idx}.jpg", "label": "b", "breed": "Bengal", "split": "train"}
        for idx in range(3)
    ] + [
        {"filename": f"val_c_{idx}.jpg", "label": "c", "breed": "Persian", "split": "val"}
        for idx in range(2)
    ] + [
        {"filename": f"val_d_{idx}.jpg", "label": "d", "breed": "Persian", "split": "val"}
        for idx in range(2)
    ]

    local = build_identity_sample(
        rows,
        SampleProfile(name="local_50pct", min_images_per_identity=2, identity_fraction=0.5, seed=7),
        split_column="split",
    )
    assert {row["split"] for row in local} == {"train", "val"}
    assert len({row["label"] for row in local if row["split"] == "train"}) == 1
    assert len({row["label"] for row in local if row["split"] == "val"}) == 1

    subset = build_experiment_subset(local, identity_fraction=0.5, split_column="split", seed=7)
    assert len({row["label"] for row in subset}) == 2

    bundle = build_local_sample_bundle(
        rows,
        local_identity_fraction=0.5,
        experiment_identity_fraction=0.5,
        split_column="split",
        seed=7,
    )
    assert bundle.identity_count == 2
    assert bundle.identity_fraction == 0.25
    assert bundle.split_counts == {"train": 3, "val": 2}
