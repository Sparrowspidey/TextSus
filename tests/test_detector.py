import pytest

from textsus.detection.detector import detect_watermark


def test_detects_watermark_when_score_equals_threshold():
    assert detect_watermark(0.5, 0.5) is True


def test_detects_watermark_when_score_is_above_threshold():
    assert detect_watermark(0.8, 0.5) is True


def test_rejects_watermark_when_score_is_below_threshold():import pytest

from textsus.detection.detector import detect_watermark


def test_detects_watermark_when_score_equals_threshold():
    assert detect_watermark(0.5, 0.5) is True


def test_detects_watermark_when_score_is_above_threshold():
    assert detect_watermark(0.8, 0.5) is True


def test_rejects_watermark_when_score_is_below_threshold():
    assert detect_watermark(0.3, 0.5) is False


def test_detector_returns_boolean():
    result = detect_watermark(0.7, 0.5)
    assert isinstance(result, bool)
    assert detect_watermark(0.3, 0.5) is False


def test_detector_returns_boolean():
    result = detect_watermark(0.7, 0.5)
    assert isinstance(result, bool)