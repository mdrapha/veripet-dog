from veripet.research.evaluation import calibrate_threshold, evaluate_verification_predictions


def test_evaluation_uses_external_threshold():
    val_labels = [1, 1, 0, 0]
    val_scores = [0.95, 0.8, 0.35, 0.1]
    calibration = calibrate_threshold(val_labels, val_scores, strategy="youden")

    test_pairs = [
        {"label": 1, "score": 0.9, "pair_type": "positive"},
        {"label": 1, "score": 0.7, "pair_type": "positive"},
        {"label": 0, "score": 0.2, "pair_type": "negative_diff_breed"},
        {"label": 0, "score": 0.6, "pair_type": "negative_same_breed"},
    ]
    result = evaluate_verification_predictions(test_pairs, threshold=calibration.threshold)

    assert 0.0 <= calibration.threshold <= 1.0
    assert result.threshold == calibration.threshold
    assert "auc" in result.metrics
    assert "negative_same_breed_accuracy" in result.metrics
