"""Features.embedding_df: numpy fast path must match the original Series.shift output exactly."""

import numpy as np
import pandas as pd
import pytest

from py3r.behaviour.features.features import Features


def _reference_embedding_df(data: pd.DataFrame, embedding: dict[str, list[int]]) -> pd.DataFrame:
    """The pre-numpy implementation, kept verbatim as the reference."""
    out = {}
    for col, shifts in embedding.items():
        for shift in shifts:
            suffix = f"t{shift:+d}" if shift != 0 else "t0"
            out[f"{col}_{suffix}"] = data[col].shift(-shift)
    return pd.DataFrame(out, index=data.index)


def _features_with(data: pd.DataFrame) -> Features:
    """embedding_df only reads self.data, so a bare Features with data set is enough."""
    f = Features.__new__(Features)
    f.data = data
    return f


@pytest.fixture(scope="module")
def features():
    n = 40
    index = pd.RangeIndex(n, name="frame")
    rng = np.random.default_rng(0)
    f64 = rng.standard_normal(n)
    f64[[3, 10, 11]] = np.nan
    f32 = rng.standard_normal(n).astype(np.float32)
    f32[5] = np.nan
    bool_na = pd.array(rng.random(n) > 0.5, dtype="boolean")
    bool_na[[2, 7]] = pd.NA
    int_na = pd.array(rng.integers(0, 5, n), dtype="Int64")
    int_na[4] = pd.NA
    float_na = pd.array(rng.standard_normal(n), dtype="Float64")
    float_na[6] = pd.NA
    cols = {
        "f64": f64,
        "f64b": rng.standard_normal(n),
        "f32": f32,
        "bool_na": bool_na,
        "int_na": int_na,
        "float_na": float_na,
        "np_bool": rng.random(n) > 0.5,
        "np_int": rng.integers(0, 5, n),
    }
    return _features_with(pd.DataFrame(cols, index=index))


EMBEDDINGS = {
    "float64 only": {"f64": [-2, -1, 0, 1, 2], "f64b": [0, 3]},
    "float32 only": {"f32": [-1, 0, 1]},
    "mixed float widths": {"f64": [-1, 0, 1], "f32": [-1, 0, 1]},
    "nullable dtypes": {"bool_na": [-1, 0, 1], "int_na": [0, 2], "float_na": [-3, 0]},
    "numpy int and bool": {"np_bool": [-1, 0, 1], "np_int": [-1, 0, 1]},
    "everything interleaved": {
        "bool_na": [-1, 1],
        "f64": [-5, 0, 5],
        "np_int": [0, 1],
        "f32": [2],
        "f64b": [-1, 0],
        "float_na": [1],
    },
    "non-contiguous and unsorted shifts": {"f64": [7, -3, 0, 2]},
    "duplicate shifts": {"f64": [1, 0, 1, -1, 0], "bool_na": [1, 1]},
    "shift beyond length": {"f64": [-10_000, 0, 10_000], "bool_na": [10_000]},
    "empty shift list": {"f64": [], "f64b": [0, 1], "bool_na": []},
    "empty embedding": {},
}


@pytest.mark.parametrize("embedding", EMBEDDINGS.values(), ids=EMBEDDINGS.keys())
def test_matches_reference(features, embedding):
    expected = _reference_embedding_df(features.data, embedding)
    result = features.embedding_df(embedding)
    pd.testing.assert_frame_equal(result, expected)


def test_nullable_edges_are_pd_na_not_nan(features):
    result = features.embedding_df({"bool_na": [1], "f64": [1]})
    assert result["bool_na_t+1"].dtype == "boolean"
    assert result["bool_na_t+1"].iloc[-1] is pd.NA
    assert result["f64_t+1"].dtype == np.float64
    assert np.isnan(result["f64_t+1"].iloc[-1])


def test_zero_rows(features):
    empty = _features_with(features.data.iloc[:0])
    embedding = {"f64": [-1, 0, 1], "bool_na": [1]}
    pd.testing.assert_frame_equal(
        empty.embedding_df(embedding), _reference_embedding_df(empty.data, embedding)
    )


def test_missing_column_raises(features):
    with pytest.raises(ValueError, match="not present"):
        features.embedding_df({"nope": [0]})
