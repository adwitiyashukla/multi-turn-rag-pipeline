import numpy as np

from turnrag.answer import (
    FEATURES,
    Gate,
    auc,
    best_threshold,
    fit_logistic,
    is_idk,
    probe_messages,
    probe_no,
)


def test_logistic_separates_and_auc():
    rng = np.random.default_rng(0)
    good = rng.normal(2, 1, size=(80, len(FEATURES)))
    bad = rng.normal(-2, 1, size=(20, len(FEATURES)))
    x = np.vstack([good, bad])
    y = np.asarray([1] * 80 + [0] * 20)
    weights, bias, mean, scale = fit_logistic(x, y)
    scores = 1 / (1 + np.exp(-(((x - mean) / scale) @ weights + bias)))
    assert auc(scores, y) > 0.99
    gate = Gate(weights.tolist(), bias, mean.tolist(), scale.tolist(), 0.7, 0.6)
    assert gate.decide(good[0])[0] == "answer"
    assert gate.decide(bad[0])[0] == "idk"
    assert gate.clarifies(0.9) and not gate.clarifies(0.1)


def test_best_threshold():
    p = np.asarray([0.1, 0.2, 0.6, 0.9])
    answer = np.asarray([0.0, 0.0, 0.5, 0.6])
    idk = np.asarray([1.0, 1.0, 0.0, 0.0])
    threshold, gain = best_threshold(p, answer, idk, np.ones(4))
    assert abs(threshold - 0.4) < 1e-9 and abs(gain - 0.5) < 1e-9
    threshold, gain = best_threshold(p, np.ones(4), np.zeros(4), np.ones(4))
    assert threshold == 0.0 and gain == 0.0
    tied = np.asarray([0.3, 0.3])
    threshold, _ = best_threshold(tied, np.asarray([0.0, 0.9]), np.asarray([1.0, 0.0]), np.ones(2))
    assert threshold in (0.0, 1.0)


def test_auc_handles_ties():
    assert auc(np.asarray([0.5, 0.5]), np.asarray([1, 0])) == 0.5


def test_idk_detection():
    text = "I do not have specific information about that in the documents I can search."
    assert is_idk(text, text)
    assert not is_idk("Juno reached Jupiter in 2016.", text)


def test_probe():
    assert abs(probe_no({"yes": 0.0, "no": 0.0, "The": 3.0}) - 0.5) < 1e-9
    assert probe_no({"No": -0.1, "Yes": -2.4}) > 0.9
    assert probe_no(None) == 0.5
    chat = [{"role": "system", "content": "s"}, {"role": "user", "content": "docs"}]
    probe = probe_messages(chat)
    assert probe[-1]["content"].startswith("docs") and chat[-1]["content"] == "docs"
