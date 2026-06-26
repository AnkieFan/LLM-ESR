"""Unit tests for the extension helper functions (no GPU / dataset needed).

Run:  python -m pytest tests/   ||   python tests/test_extensions.py
"""
import os, sys, tempfile, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _args(**kw):
    a = types.SimpleNamespace(emb_path="", dataset="yelp", d_llm=0, sampling_method="topn", sim_pool="mean")
    a.__dict__.update(kw)
    return a


def test_sample_sim_topn_is_first_k():
    from generators.data import sample_sim
    pool = list(range(100))
    assert sample_sim("topn", pool, 10) == pool[:10]


def test_sample_sim_stochastic_draws_from_top50():
    from generators.data import sample_sim
    pool = list(range(100))
    s = sample_sim("stochastic", pool, 10)
    assert len(s) == 10 and set(s) <= set(pool[:50])      # E11/E12: k random from the top-50 pool


def test_pool_sim_mean_matches_torch_mean():
    import torch
    from models.LLMESR import pool_sim
    lf, slf = torch.randn(4, 8), torch.randn(4, 3, 8)
    assert pool_sim("mean", lf, slf).shape == (4, 8)
    assert torch.allclose(pool_sim("mean", lf, slf), slf.mean(1))


def test_pool_sim_weighted_shape_and_normalisation():
    import torch
    from models.LLMESR import pool_sim
    lf, slf = torch.randn(4, 8), torch.randn(4, 3, 8)
    out = pool_sim("weighted", lf, slf)                   # E10: similarity-weighted teacher pooling
    assert out.shape == (4, 8)


def test_load_llm_item_emb_reads_npy_and_asserts_dim():
    from models.DualLLMSRS import load_llm_item_emb
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "item.npy"), np.random.randn(50, 3072).astype("float32"))
        got = load_llm_item_emb(_args(emb_path=d, d_llm=3072))      # E25: encoder swap
        assert got.shape == (50, 3072) and got.dtype == np.float32
        try:
            load_llm_item_emb(_args(emb_path=d, d_llm=1536))
            raise AssertionError("expected a dim-mismatch ValueError")
        except ValueError:
            pass


def test_load_id_item_emb_prefers_run_local_pca():
    from models.DualLLMSRS import load_id_item_emb
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "pca64.npy"), np.random.randn(50, 64).astype("float32"))
        np.save(os.path.join(d, "item.npy"), np.zeros((50, 1536), "float32"))
        assert load_id_item_emb(_args(emb_path=d)).shape == (50, 64)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except Exception as e:
            fails += 1; print("FAIL", fn.__name__, "::", repr(e))
    print("\n%d/%d passed" % (len(fns) - fails, len(fns)))
    sys.exit(1 if fails else 0)
