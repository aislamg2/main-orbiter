"""Hardware-free tests for live classification (no camera, no Hailo)."""

import numpy as np
import pytest

from pantilt import load_config
from pantilt.classify import CLASS_NAMES, LiveClassifier, preprocess
from pantilt.cli import build_parser


def test_preprocess_shape_and_dtype():
    frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 4), dtype=np.uint8)
    x = preprocess(frame)
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.uint8
    assert x.flags.writeable and x.flags.c_contiguous   # HailoRT requirement
    assert (x[..., 0] == x[..., 1]).all() and (x[..., 1] == x[..., 2]).all()  # grayscale


def test_preprocess_takes_centered_square():
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    frame[:, 100:200] = 255          # only the centered 100 x 100 square is white
    assert (preprocess(frame) == 255).all()
    frame[40:60, 140:160] = 0        # a dark spot dead centre fills a 0.2 crop
    assert (preprocess(frame, crop=0.2) == 0).all()


def test_preprocess_rejects_bad_input():
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10, 3), dtype=np.uint8), crop=0)


def onehot(i):
    p = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    p[i] = 1
    return p


def test_live_classifier_smooths_and_labels():
    seq = iter([onehot(1)] * 3 + [onehot(2)] * 2)
    live = LiveClassifier(lambda: np.zeros((50, 50, 3), np.uint8), lambda x: next(seq), smooth=5)
    for _ in range(5):
        p = live.step()
    assert p.label == "crater"                       # 3 of 5 frames
    assert p.confidence == pytest.approx(0.6)
    assert p.top3[1] == ("dark dune", pytest.approx(0.4))
    assert p.frames == 5
    assert live.status()["class"] == "crater"


def test_live_classifier_min_confidence():
    live = LiveClassifier(lambda: np.zeros((50, 50, 3), np.uint8),
                          lambda x: np.full(len(CLASS_NAMES), 1 / len(CLASS_NAMES)),
                          min_confidence=0.5)
    assert live.step().label == "uncertain"


def test_live_thread_reports_errors():
    def boom():
        raise RuntimeError("camera gone")

    live = LiveClassifier(boom, lambda x: onehot(0)).start()
    live.stop()
    assert isinstance(live.error, RuntimeError)


def test_config_has_model_section():
    cfg = load_config("config.toml")
    assert cfg.model.hef_path.endswith(".hef")
    assert cfg.model.smooth_frames >= 1


def test_live_command_parses():
    args = build_parser().parse_args(["live", "--scan", "--cols", "3", "--log", "x.csv"])
    assert args.func.__name__ == "cmd_live" and args.cols == 3
