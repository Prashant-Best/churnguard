import numpy as np
import pytest

from churnguard.evaluate import choose_threshold, classification_metrics, gain_at

Y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
P = np.array([0.1, 0.2, 0.15, 0.3, 0.4, 0.7, 0.35, 0.6, 0.8, 0.9])


def test_metrics_confusion_counts_add_up():
    m = classification_metrics(Y, P, threshold=0.5)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(Y)
    assert 0 <= m["roc_auc"] <= 1


def test_recall_strategy_meets_target():
    t = choose_threshold(Y, P, strategy="recall", min_recall=0.75)
    recall = ((P >= t) & (Y == 1)).sum() / Y.sum()
    assert recall >= 0.75


def test_f1_strategy_returns_valid_threshold():
    t = choose_threshold(Y, P, strategy="f1")
    assert P.min() <= t <= P.max()


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        choose_threshold(Y, P, strategy="magic")


def test_gain_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    perfect = np.linspace(1, 0, 10)
    assert gain_at(y, perfect, 0.2) == 1.0


def test_gain_random_ranking_is_bounded():
    assert 0.0 <= gain_at(Y, P, 0.3) <= 1.0
