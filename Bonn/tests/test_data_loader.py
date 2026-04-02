"""Unit tests for data_loader module."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_loader import (
    load_single_file,
    sliding_window_segment,
    load_recordings,
    segment_and_normalize,
)
from config import POINTS_PER_FILE, WINDOW_SIZE, WINDOW_STRIDE, DATA_DIR


SAMPLE_FILE = os.path.join(DATA_DIR, "Z", "Z001.txt")
SAMPLE_FILE_EXISTS = os.path.isfile(SAMPLE_FILE)


@pytest.mark.skipif(not SAMPLE_FILE_EXISTS, reason="Data file Z001.txt not found")
class TestLoadSingleFile:
    def test_returns_1d_array(self):
        sig = load_single_file(SAMPLE_FILE)
        assert sig.ndim == 1

    def test_correct_length(self):
        sig = load_single_file(SAMPLE_FILE)
        assert len(sig) == POINTS_PER_FILE

    def test_dtype_float32(self):
        sig = load_single_file(SAMPLE_FILE)
        assert sig.dtype == np.float32


class TestSlidingWindowSegment:
    def test_output_shape(self):
        signal = np.random.randn(4097).astype(np.float32)
        segs = sliding_window_segment(signal, window_size=1024, stride=512)
        expected_num = (4097 - 1024) // 512 + 1
        assert segs.shape == (expected_num, 1024)

    def test_window_values_match_original(self):
        signal = np.arange(4097, dtype=np.float32)
        segs = sliding_window_segment(signal, window_size=1024, stride=512)
        np.testing.assert_array_equal(segs[0], signal[:1024])
        np.testing.assert_array_equal(segs[1], signal[512:512 + 1024])

    def test_single_window_when_equal_length(self):
        signal = np.random.randn(1024).astype(np.float32)
        segs = sliding_window_segment(signal, window_size=1024, stride=512)
        assert segs.shape == (1, 1024)

    def test_no_window_when_too_short(self):
        signal = np.random.randn(500).astype(np.float32)
        segs = sliding_window_segment(signal, window_size=1024, stride=512)
        assert segs.shape[0] == 0


@pytest.mark.skipif(not SAMPLE_FILE_EXISTS, reason="Data files not found")
class TestLoadRecordings:
    @pytest.mark.parametrize("task,expected_classes", [
        ("binary", 2),
        ("three", 3),
        ("five", 5),
    ])
    def test_class_count(self, task, expected_classes):
        recordings, labels, class_names = load_recordings(task=task)
        assert len(class_names) == expected_classes

    def test_binary_labels_are_0_or_1(self):
        _, labels, _ = load_recordings(task="binary")
        assert set(np.unique(labels)).issubset({0, 1})

    def test_five_class_labels(self):
        _, labels, _ = load_recordings(task="five")
        assert set(np.unique(labels)) == {0, 1, 2, 3, 4}

    def test_recordings_are_correct_length(self):
        recordings, _, _ = load_recordings(task="binary")
        for rec in recordings:
            assert len(rec) == POINTS_PER_FILE

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            load_recordings(task="invalid")


@pytest.mark.skipif(not SAMPLE_FILE_EXISTS, reason="Data files not found")
class TestSegmentAndNormalize:
    def _get_split_data(self):
        recordings, labels, _ = load_recordings(task="binary")
        mid = len(recordings) // 2
        return (
            recordings[:mid], labels[:mid],
            recordings[mid:], labels[mid:],
        )

    def test_output_shapes(self):
        train_recs, train_lbls, val_recs, val_lbls = self._get_split_data()
        X_train, y_train, X_val, y_val = segment_and_normalize(
            train_recs, train_lbls, val_recs, val_lbls,
        )
        assert X_train.ndim == 3
        assert X_train.shape[1] == WINDOW_SIZE
        assert X_train.shape[2] == 1
        assert len(y_train) == X_train.shape[0]
        assert X_val.shape[1] == WINDOW_SIZE

    def test_train_is_roughly_standardized(self):
        """Training data should be approximately zero-mean after standardization."""
        train_recs, train_lbls, val_recs, val_lbls = self._get_split_data()
        X_train, _, _, _ = segment_and_normalize(
            train_recs, train_lbls, val_recs, val_lbls,
        )
        mean_val = np.abs(X_train.mean())
        assert mean_val < 0.1, f"Mean too far from 0: {mean_val}"
