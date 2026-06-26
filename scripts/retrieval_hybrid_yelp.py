import os, pickle
import numpy as np
from collections import defaultdict

handled = "./data/yelp/handled"
N = 50  # rerank the top 50 llm candidates

sim = pickle.load(open(handled + "/sim_user_100.pkl", "rb"))

items = defaultdict(set)
for line in open(handled + "/inter.txt"):
    u, i = line.split()
    items[int(u)].add(int(i))

out = np.zeros_like(sim)
for u in range(len(sim)):
    cand = sim[u][:N]
    shared = [len(items[u] & items[v]) for v in cand]
    order = np.argsort(-np.array(shared), kind="stable")   # most overlap first, llm order on ties
    out[u] = np.concatenate([cand[order], sim[u][N:]])

pickle.dump(out, open(handled + "/sim_user_hybrid_100.pkl", "wb"))
print("done", out.shape)
