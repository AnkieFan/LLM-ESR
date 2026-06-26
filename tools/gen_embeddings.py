#!/usr/bin/env python
"""Generate frozen LLM embeddings for LLM-ESR via an OpenAI-compatible gateway.

This is the scriptable equivalent of the repo's
``data/<dataset>/get_item_embedding.ipynb`` and ``get_user_embedding.ipynb``
notebooks: it builds the *exact same* text prompts (same templates, same field
order, same length caps) and embeds them, but lets you pick any embedding model
and target dimension so the only thing that changes between ablation runs is the
``embeddings/<dataset>/<run_id>/`` folder that training loads.

Runs on the LOGIN NODE (needs internet for the gateway, API-bound, no GPU).
Requires env vars OPENAI_API_KEY and OPENAI_BASE_URL.

Outputs (np.float32):
    <out_root>/<dataset>/<run_id>/item.npy   # row i  == item id (i+1)
    <out_root>/<dataset>/<run_id>/user.npy   # notebook user order (see below)

Item rows are ordered by item id 1..N (via id_map["id2item"]) so they line up
with handled/inter.txt and pca64_itm_emb_np.pkl. User rows follow the notebook
order (first-appearance order of users in inter.txt), matching usr_emb_np.pkl.
"""

import os
import sys
import json
import copy
import time
import argparse
from collections import defaultdict

import numpy as np


# prompt construction: ports of the data/<dataset>/*.ipynb cells

def _yelp_item_prompt(value):
    template = ("The point of interest has following attributes: \n name is "
                "<NAME>; category is <CATEGORY>; type is <TYPE>; open status is "
                "<OPEN>; review count is <COUNT>; city is <CITY>; average score "
                "is <STARS>.")
    s = copy.deepcopy(template)
    s = s.replace("<NAME>", str(value.get("name", "unknown")))
    cate_str = ""
    cats = value.get("categories", "")
    if cats is None:
        cats = []
    elif isinstance(cats, str):
        cats = [c.strip() for c in cats.split(",")] if cats else []
    for cate in cats:
        cate_str += (cate + " ")
    s = s.replace("<CATEGORY>", cate_str)
    s = s.replace("<TYPE>", str(value.get("type", "business")))
    s = s.replace("<OPEN>", str(value.get("open", value.get("is_open", "unknown"))))
    s = s.replace("<COUNT>", str(value.get("review_count", "unknown")))
    s = s.replace("<CITY>", str(value.get("city", "unknown")))
    s = s.replace("<STARS>", str(value.get("stars", "unknown")))
    return s


def _amazon_item_prompt(value, dataset):
    # beauty / fashion share the same structure with slightly different wording.
    if dataset == "beauty":
        template = ("The beauty item has following attributes: \n name is "
                    "<TITLE>; brand is <BRAND>; price is <PRICE>. \n")
        feat_key, feat_nested = "categories", True
    else:  # fashion
        template = ("The fashion item has following attributes: \n name is "
                    "<TITLE>; brand is <BRAND>; score is <DATE>; price is "
                    "<PRICE>. \n")
        feat_key, feat_nested = "feature", False
    feat_template = "The item has following features: <FEATURE>. \n"
    desc_template = "The item has following descriptions: <DESCRIPTION>. \n"

    def get_attri(item_str, attri, info):
        val = info.get(attri, None)
        if val is None or (isinstance(val, str) and len(val) > 100):
            return item_str.replace(f"<{attri.upper()}>", "unknown")
        return item_str.replace(f"<{attri.upper()}>", str(val))

    def get_feat(item_str, feat, info):
        if feat not in info:
            return ""
        seq = info[feat]
        assert isinstance(seq, list)
        items = seq[0] if (feat_nested and seq and isinstance(seq[0], list)) else seq
        feat_str = ""
        for meta_feat in items:
            feat_str = feat_str + str(meta_feat) + "; "
        new_str = item_str.replace(f"<{feat.upper()}>", feat_str)
        if len(new_str) > 2048:
            return new_str[:2048]
        return new_str

    s = copy.deepcopy(template)
    s = get_attri(s, "title", value)
    s = get_attri(s, "brand", value)
    s = get_attri(s, "date", value)
    s = get_attri(s, "price", value)
    feat_str = get_feat(copy.deepcopy(feat_template), feat_key, value)
    if dataset == "beauty":
        desc_str = get_attri(copy.deepcopy(desc_template), "description", value)
    else:
        desc_str = get_feat(copy.deepcopy(desc_template), "description", value)
    return s + feat_str + desc_str


def _item_name(dataset, info):
    if dataset == "yelp":
        return info.get("name")
    return info.get("title")


_USER_TEMPLATES = {
    "yelp": ("The user has visited following point of interests: \n<HISTORY> "
             "\nplease conclude the user's perference."),
    "beauty": ("The user has visited following fashions: \n<HISTORY> \nplease "
               "conclude the user's perference."),
    "fashion": ("The user has visited following fashions: \n<HISTORY> \nplease "
                "conclude the user's perference."),
}


def load_item_prompts(handled_dir, dataset):
    """Build item prompts ordered by item id 1..N (row i -> item id i+1)."""
    item2attr = json.load(open(os.path.join(handled_dir, "item2attributes.json")))
    id_map = json.load(open(os.path.join(handled_dir, "id_map.json")))
    id2item = id_map["id2item"]

    # item2attributes may be keyed by raw id (yelp) or by numeric item id
    # (amazon). Probe with the first id2item entry.
    sample_key = id2item["1"]
    keyed_by_raw = sample_key in item2attr

    prompts = []
    n = len(id2item)
    for i in range(1, n + 1):
        raw = id2item[str(i)]
        if keyed_by_raw:
            info = item2attr.get(raw, {})
        else:
            info = item2attr.get(str(i), {})
        if dataset == "yelp":
            prompts.append(_yelp_item_prompt(info))
        else:
            prompts.append(_amazon_item_prompt(info, dataset))
    return prompts


def _load_train_histories(handled_dir):
    """Replicate the notebooks' load_dataset(): user_train in first-appearance
    order of users in inter.txt (== usr_emb_np.pkl row order)."""
    User = defaultdict(list)
    with open(os.path.join(handled_dir, "inter.txt")) as f:
        for line in f:
            u, i = line.rstrip().split(" ")
            User[int(u)].append(int(i))
    train = {}
    for user in User:
        if len(User[user]) < 3:
            train[user] = User[user]
        else:
            train[user] = User[user][:-2]
    return train


def load_user_prompts(handled_dir, dataset):
    """Build user prompts in notebook order (first-appearance order of users)."""
    item2attr = json.load(open(os.path.join(handled_dir, "item2attributes.json")))
    id_map = json.load(open(os.path.join(handled_dir, "id_map.json")))
    id2item = id_map["id2item"]
    sample_key = id2item["1"]
    keyed_by_raw = sample_key in item2attr

    template = _USER_TEMPLATES[dataset]
    histories = _load_train_histories(handled_dir)

    prompts = []
    for user, history in histories.items():
        s = copy.deepcopy(template)
        hist_str = ""
        for item in history:
            raw = id2item[str(item)]
            info = item2attr.get(raw, {}) if keyed_by_raw else item2attr.get(str(item), {})
            name = _item_name(dataset, info)
            if name is None:
                continue
            hist_str = hist_str + str(name) + ", "
        if len(hist_str) > 8000:
            hist_str = hist_str[-8000:]
        prompts.append(s.replace("<HISTORY>", hist_str))
    return prompts


# embedding via the OpenAI-compatible gateway

def _make_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY is not set (needed for the gateway).")
    if not base_url:
        sys.exit("ERROR: OPENAI_BASE_URL is not set (needed for the gateway).")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("ERROR: the `openai` package is not installed. On the login "
                 "node run: pip install openai")
    return OpenAI(api_key=api_key, base_url=base_url)


def _save_ckpt(cache_path, out, i):
    """Atomically persist the first i completed rows (so a crash mid-write
    never corrupts the resume file)."""
    tmp = cache_path + ".tmp.npy"
    np.save(tmp, out[:i])
    os.replace(tmp, cache_path)


def embed_prompts(client, prompts, model, dimensions, batch_size, cache_path,
                  checkpoint_every=20, expected_dim=None, max_retries=8):
    """Embed prompts in batches with intra-file checkpointing.

    Every ``checkpoint_every`` batches the completed rows are flushed to
    ``cache_path`` (atomically). Re-running resumes from the last flushed row,
    so an interrupted run loses at most ``checkpoint_every`` batches of work.
    A cache whose dimension does not match ``expected_dim`` (e.g. left over from
    a different model/--dimensions setting) is discarded rather than resumed.
    """
    n = len(prompts)
    dim = None
    out = None
    start = 0

    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if expected_dim and cached.shape[1] != expected_dim:
            print(f"  ignoring stale cache (dim {cached.shape[1]} != expected "
                  f"{expected_dim}); regenerating from scratch")
        elif cached.shape[0] >= n:
            return cached[:n]
        else:
            out = np.zeros((n, cached.shape[1]), dtype=np.float32)
            out[:cached.shape[0]] = cached
            dim = cached.shape[1]
            start = cached.shape[0]
            print(f"  resuming from cache at row {start}/{n}")

    i = start
    batches_done = 0
    while i < n:
        batch = prompts[i:i + batch_size]
        # blank prompts are not allowed by the API; substitute a space.
        batch = [p if p.strip() else " " for p in batch]
        kwargs = {"model": model, "input": batch}
        if dimensions:
            kwargs["dimensions"] = dimensions
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(**kwargs)
                break
            except Exception as e:  # noqa: BLE001 - gateway/transient errors
                wait = min(2 ** attempt, 60)
                print(f"  batch {i}: {type(e).__name__}: {e}; retry in {wait}s")
                time.sleep(wait)
        else:
            sys.exit(f"ERROR: batch starting at {i} failed after retries.")

        vecs = [d.embedding for d in resp.data]
        if out is None:
            dim = len(vecs[0])
            out = np.zeros((n, dim), dtype=np.float32)
        out[i:i + len(vecs)] = np.asarray(vecs, dtype=np.float32)
        i += len(vecs)
        batches_done += 1
        print(f"  {i}/{n}", end="\r", flush=True)

        if cache_path and (batches_done % checkpoint_every == 0 or i >= n):
            _save_ckpt(cache_path, out, i)
    print()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="yelp",
                    choices=["yelp", "fashion", "beauty"])
    ap.add_argument("--model", required=True,
                    help="embedding model id, e.g. text-embedding-3-small")
    ap.add_argument("--run-id", required=True,
                    help="output subfolder name, e.g. small / large / ada")
    ap.add_argument("--dimensions", type=int, default=None,
                    help="target dim sent to the API (text-embedding-3-* only; "
                         "use to truncate large to 1536). Omit for ada-002.")
    ap.add_argument("--d-llm", type=int, default=None,
                    help="assert the produced embedding dimension equals this.")
    ap.add_argument("--handled-dir", default=None,
                    help="override data/<dataset>/handled directory.")
    ap.add_argument("--out-root", default="embeddings",
                    help="root output dir (default: embeddings/).")
    ap.add_argument("--kind", default="both",
                    choices=["item", "user", "both"])
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="prompts per request (gateway hard max is 2048). ")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="flush progress to the resume cache every N batches.")
    ap.add_argument("--limit", type=int, default=None,
                    help="embed only the first N prompts (smoke testing only; "
                         "produces an incomplete file, not for training).")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output .npy already exists.")
    args = ap.parse_args()

    handled = args.handled_dir or os.path.join("data", args.dataset, "handled")
    for fn in ("id_map.json", "item2attributes.json", "inter.txt"):
        if not os.path.exists(os.path.join(handled, fn)):
            sys.exit(f"ERROR: missing prompt source {os.path.join(handled, fn)}.\n"
                     f"       These are produced by data/data_process.py + "
                     f"convert_inter.ipynb. gen_embeddings replicates the "
                     f"notebooks and needs them to build prompts.")

    out_dir = os.path.join(args.out_root, args.dataset, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    kinds = ["item", "user"] if args.kind == "both" else [args.kind]
    client = None

    for kind in kinds:
        out_path = os.path.join(out_dir, f"{kind}.npy")
        if os.path.exists(out_path) and not args.overwrite:
            arr = np.load(out_path)
            print(f"[{kind}] exists: {out_path} {arr.shape} (skip)")
            continue

        print(f"[{kind}] building prompts from {handled} ...")
        if kind == "item":
            prompts = load_item_prompts(handled, args.dataset)
        else:
            prompts = load_user_prompts(handled, args.dataset)
        if args.limit:
            prompts = prompts[:args.limit]
        print(f"[{kind}] {len(prompts)} prompts; sample:\n  {prompts[0][:200]!r}")

        if client is None:
            client = _make_client()

        cache_path = os.path.join(cache_dir, f"{kind}.npy")
        emb = embed_prompts(client, prompts, args.model, args.dimensions,
                            args.batch_size, cache_path,
                            checkpoint_every=args.checkpoint_every,
                            expected_dim=args.d_llm)

        if args.d_llm and emb.shape[1] != args.d_llm:
            sys.exit(f"ERROR: produced dim {emb.shape[1]} != --d-llm {args.d_llm}. "
                     f"For text-embedding-3-large pass --dimensions {args.d_llm}.")

        np.save(out_path, emb.astype(np.float32))
        print(f"[{kind}] saved {out_path} {emb.shape}")

    print("done.")


if __name__ == "__main__":
    main()
