"""Quick checks that don't need the datasets: run with `pytest`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from plankton.data import class_names, level1_to_level2, write_predictions
from plankton.images import pad_to_square

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate  # noqa: E402


def test_class_lists():
    assert len(class_names(1)) == 54
    assert len(class_names(2)) == 32
    assert level1_to_level2()["like<Copepoda"] == "Copepoda"


def test_pad_to_square_uses_background():
    img = Image.fromarray(np.full((10, 30), 200, dtype=np.uint8))
    out = pad_to_square(img)
    assert out.size == (30, 30)
    assert np.asarray(out)[0, 0] == 200  # padded with the image's own background, not white


def test_write_predictions_roundtrip(tmp_path):
    path = write_predictions(["a", "b"], ["detritus", "unknown"], tmp_path / "p.csv")
    assert pd.read_csv(path).to_dict("list") == {"object_id": ["a", "b"], "label": ["detritus", "unknown"]}
    with pytest.raises(ValueError):
        write_predictions(["a", "a"], ["x", "y"], tmp_path / "dup.csv")


def test_evaluate_perfect_and_open_set():
    classes = evaluate.load_classes()
    truth = pd.DataFrame({"object_id": ["1", "2", "3", "4"],
                          "label": ["detritus", "Copepoda", "Copepoda", "unknown"]})
    pred = evaluate.clean_predictions(
        pd.DataFrame({"object_id": ["1", "2", "3", "4"],
                      "label": ["detritus", "like<Copepoda", "Copepoda", "unknown"]}), 2, classes)
    m = evaluate.evaluate(truth, pred, 2, classes)["metrics"]["unweighted"]
    assert m["macro_f1"] == pytest.approx(1.0)  # level-1 name mapped to level 2
    assert m["unknown_recall"] == pytest.approx(1.0)


def test_evaluate_rejects_bad_labels():
    classes = evaluate.load_classes()
    with pytest.raises(evaluate.SubmissionError):
        evaluate.clean_predictions(pd.DataFrame({"object_id": ["1"], "label": ["banana"]}), 2, classes)
