"""Unit tests for model module."""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import build_model, count_parameters, CNNBiLSTMAttention, BiLSTMAttention, Pure1DCNN
from config import WINDOW_SIZE


DEVICE = torch.device("cpu")
BATCH_SIZE = 4


def _make_dummy_input(batch_size: int = BATCH_SIZE, seq_len: int = WINDOW_SIZE):
    return torch.randn(batch_size, seq_len, 1)


class TestBuildModel:
    def test_hybrid(self):
        model = build_model("hybrid", num_classes=2)
        assert isinstance(model, CNNBiLSTMAttention)

    def test_bilstm(self):
        model = build_model("bilstm", num_classes=2)
        assert isinstance(model, BiLSTMAttention)

    def test_cnn(self):
        model = build_model("cnn", num_classes=2)
        assert isinstance(model, Pure1DCNN)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            build_model("transformer", num_classes=2)


class TestCNNBiLSTMAttention:
    @pytest.mark.parametrize("num_classes", [2, 3, 5])
    def test_output_shape(self, num_classes):
        model = build_model("hybrid", num_classes=num_classes).to(DEVICE)
        x = _make_dummy_input()
        logits, attn = model(x)
        if num_classes == 2:
            assert logits.shape == (BATCH_SIZE,)
        else:
            assert logits.shape == (BATCH_SIZE, num_classes)

    def test_attention_weights_returned(self):
        model = build_model("hybrid", num_classes=2).to(DEVICE)
        _, attn = model(_make_dummy_input())
        assert attn is not None
        assert attn.ndim == 2
        assert attn.shape[0] == BATCH_SIZE

    def test_attention_weights_sum_to_one(self):
        model = build_model("hybrid", num_classes=2).to(DEVICE)
        _, attn = model(_make_dummy_input())
        sums = attn.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(BATCH_SIZE), atol=1e-5, rtol=1e-5)

    def test_has_trainable_params(self):
        model = build_model("hybrid", num_classes=2)
        assert count_parameters(model) > 0


class TestBiLSTMAttention:
    @pytest.mark.parametrize("num_classes", [2, 5])
    def test_output_shape(self, num_classes):
        model = build_model("bilstm", num_classes=num_classes).to(DEVICE)
        logits, attn = model(_make_dummy_input())
        if num_classes == 2:
            assert logits.shape == (BATCH_SIZE,)
        else:
            assert logits.shape == (BATCH_SIZE, num_classes)
        assert attn is not None

    def test_different_from_hybrid_param_count(self):
        hybrid = build_model("hybrid", num_classes=2)
        bilstm = build_model("bilstm", num_classes=2)
        assert count_parameters(hybrid) != count_parameters(bilstm)


class TestPure1DCNN:
    @pytest.mark.parametrize("num_classes", [2, 3, 5])
    def test_output_shape(self, num_classes):
        model = build_model("cnn", num_classes=num_classes).to(DEVICE)
        logits, attn = model(_make_dummy_input())
        if num_classes == 2:
            assert logits.shape == (BATCH_SIZE,)
        else:
            assert logits.shape == (BATCH_SIZE, num_classes)

    def test_no_attention_weights(self):
        model = build_model("cnn", num_classes=2).to(DEVICE)
        _, attn = model(_make_dummy_input())
        assert attn is None
