"""Sequence-recommendation trainer.

Beyond standard train/eval, ``SeqTrainer.eval()`` hosts the inference-time extensions as flag-gated
post-loop sweeps over the frozen model: popularity debiasing (``--pop_debias``), D6/D7 user-semantic
scoring (``--usr_sem``/``--usr_sem_d7``/``--usr_sem_seg``), and two-specialist routing (``--moe_diag``).
Each writes a ``*.csv`` sweep next to the checkpoint; ``scripts/aggregate_*.py`` reduce them to tables.
See ``docs/CODE_OVERVIEW.md`` for the full map. Defaults reproduce the original LLM-ESR.
"""
# here put the import lib
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from utils.utils import metric_report, metric_len_report, record_csv, metric_pop_report
from utils.utils import metric_len_5group, metric_pop_5group
from utils.utils import per_user_ndcg_hr


class SeqTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            inputs = self._prepare_train_inputs(batch)
            loss = self.model(**inputs)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader

        self.model.eval()
        if test and getattr(self.args, "full_rank", False):
            return self._eval_fullrank(test_loader, desc)
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        _dbg = test and getattr(self.args, "pop_debias", False)
        _usr = test and getattr(self.args, "usr_sem", False)
        _d7 = _usr and getattr(self.args, "usr_sem_d7", False)   # D7: recency/max profiles + MoE ensemble
        _moe = test and getattr(self.args, "moe_diag", False)    # Stage-1: two-specialist diagnostic (SASRec)
        _acc = _dbg or _usr
        _dbg_S, _dbg_C, _usr_TERM, _usr_TERM_rec, _usr_TERM_max = [], [], [], [], []
        _moe_col, _moe_sem, _moe_max, _moe_cand = [], [], [], []

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])

            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                _raw = self.model.predict(**inputs)   # (B, n_cand) score, higher=better
                pred_logits = -_raw

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])
                if _acc:
                    _dbg_S.append(_raw); _dbg_C.append(inputs["item_indices"])
                if _moe:
                    H = self.model.hidden_size
                    ff = self._final_feat(inputs)                                             # (B, 2H) = [collab|sem]
                    ie = self.model._get_embedding(inputs["item_indices"])                    # (B, n_cand, 2H)
                    _moe_col.append((ie[..., :H] * ff[:, :H].unsqueeze(1)).sum(-1))            # collaborative-view score
                    _moe_sem.append((ie[..., H:] * ff[:, H:].unsqueeze(1)).sum(-1))           # semantic-view score
                    _le = self.model.llm_item_emb; _hh = _le(inputs["seq"]); _mm = (inputs["seq"] > 0)
                    _cc = _le(inputs["item_indices"])
                    _ss = torch.bmm(_cc, _hh.transpose(1, 2)).masked_fill((_mm == 0).unsqueeze(1), float("-inf"))
                    _moe_max.append(torch.nan_to_num(_ss.max(-1).values, neginf=0.0))          # max-pool content term
                    _moe_cand.append(inputs["item_indices"])                                   # per-candidate ids (for routing)
                if _usr:
                    _le = self.model.llm_item_emb
                    _sq = inputs["seq"]; _h = _le(_sq); _m = (_sq > 0).float().unsqueeze(-1)
                    _cand = _le(inputs["item_indices"])                           # (B, n_cand, d)
                    _us = (_h * _m).sum(1) / _m.sum(1).clamp(min=1.0)              # (B, d) mean (D6) centroid
                    _usr_TERM.append((_cand * _us.unsqueeze(1)).sum(-1))          # (B, n_cand)
                    if _d7:
                        _L = _h.shape[1]
                        _w = (0.9 ** (_L - 1 - torch.arange(_L, device=_h.device).float())).view(1, _L, 1) * _m
                        _ur = (_h * _w).sum(1) / _w.sum(1).clamp(min=1e-6)        # recency-weighted centroid
                        _usr_TERM_rec.append((_cand * _ur.unsqueeze(1)).sum(-1))
                        _sims = torch.bmm(_cand, _h.transpose(1, 2))              # (B, n_cand, L) cand vs each hist item
                        _sims = _sims.masked_fill((_m.squeeze(-1) == 0).unsqueeze(1), float("-inf"))
                        _mx = _sims.max(-1).values                               # (B, n_cand) max-pool (multi-interest)
                        # empty-history users (no valid position) get the max of an empty set = -inf; set to 0 (neutral, like the
                        # mean term) so that 0*term doesn't become NaN and corrupt the gamma=0 baseline.
                        _usr_TERM_max.append(torch.nan_to_num(_mx, neginf=0.0))

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))

        if test:
            self.logger.info("User Group Performance:")
            for k, v in res_len_dict.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
            self.logger.info("Item Group Performance:")
            for k, v in res_pop_dict.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict, **res_pop_dict}

        if test:
            record_csv(self.args, res_dict)

            # per-user scores for downstream paired t-tests, written to
            # <output_dir>/per_user_scores/{ndcg,hr}_s{seed}.npy, shape [N_users] float64
            per_ndcg, per_hr = per_user_ndcg_hr(pred_rank.detach().cpu().numpy())
            score_dir = os.path.join(self.args.output_dir, "per_user_scores")
            os.makedirs(score_dir, exist_ok=True)
            np.save(os.path.join(score_dir, f"ndcg_s{self.args.seed}.npy"), per_ndcg)
            np.save(os.path.join(score_dir, f"hr_s{self.args.seed}.npy"),   per_hr)
            self.logger.info(f"Per-user scores saved to {score_dir}")

        if _dbg:
            S = torch.cat(_dbg_S, dim=0)
            C = torch.cat(_dbg_C, dim=0).long()
            _lp = torch.log1p(torch.tensor(np.asarray(self.item_pop), dtype=torch.float32, device=S.device))
            C = C.clamp(0, _lp.shape[0] - 1)
            CP = _lp[C]
            _tgt = target_items.detach().cpu().numpy()
            self.logger.info("Popularity-debias sweep (score -= alpha*log1p(pop)):")
            _rows = ["alpha,HR@10,NDCG@10,Tail_HR@10,Tail_NDCG@10"]
            for a in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
                adj = S - a * CP
                rk = (adj > adj[:, 0:1]).sum(dim=1).detach().cpu().numpy().astype(float)
                r = metric_report(rk)
                rp = metric_pop_report(rk, self.item_pop, _tgt, args=self.args)
                th = rp.get("Tail HR@10", float("nan")); tn = rp.get("Tail NDCG@10", float("nan"))
                self.logger.info("  alpha=%.2f  HR@10=%.4f  NDCG@10=%.4f  Tail_HR@10=%.4f" % (a, r["HR@10"], r["NDCG@10"], th))
                _rows.append("%.2f,%.5f,%.5f,%.5f,%.5f" % (a, r["HR@10"], r["NDCG@10"], th, tn))
            with open(os.path.join(self.args.output_dir, "debias_sweep.csv"), "w") as _f:
                _f.write("\n".join(_rows) + "\n")

        if _usr:
            S = torch.cat(_dbg_S, dim=0)
            TERM = torch.cat(_usr_TERM, dim=0)
            _tgt = target_items.detach().cpu().numpy()
            _sl = seq_len.detach().cpu().numpy()
            # full Table-1-style breakdown per gamma (paper's groups). Columns kept backward-compatible:
            # HR@10/NDCG@10 = Overall, Tail_* = Tail-Item, then Head-Item / Tail-User / Head-User (H@10 & N@10).
            FULL_HDR = ("gamma,HR@10,NDCG@10,Tail_HR@10,Tail_NDCG@10,HeadItem_HR@10,HeadItem_NDCG@10,"
                        "TailUser_HR@10,TailUser_NDCG@10,HeadUser_HR@10,HeadUser_NDCG@10")

            def _full_row(g_val, rk):
                r = metric_report(rk)
                rp = metric_pop_report(rk, self.item_pop, _tgt, args=self.args)   # Tail/Popular = Tail/Head Item
                rl = metric_len_report(rk, _sl, aug_len=self.args.aug_seq_len, args=self.args)  # Short/Long = Tail/Head User
                nan = float("nan")
                return "%.1f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f" % (
                    g_val, r["HR@10"], r["NDCG@10"],
                    rp.get("Tail HR@10", nan), rp.get("Tail NDCG@10", nan),
                    rp.get("Popular HR@10", nan), rp.get("Popular NDCG@10", nan),
                    rl.get("Short HR@10", nan), rl.get("Short NDCG@10", nan),
                    rl.get("Long HR@10", nan), rl.get("Long NDCG@10", nan))
            self.logger.info("User-semantic scoring sweep (score += gamma * <user_hist_sem, item_llm>):")
            _rows = [FULL_HDR]
            for g in [0.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
                adj = S + g * TERM
                rk = (adj > adj[:, 0:1]).sum(dim=1).detach().cpu().numpy().astype(float)
                _rows.append(_full_row(g, rk))
            with open(os.path.join(self.args.output_dir, "usr_sem_sweep.csv"), "w") as _f:
                _f.write("\n".join(_rows) + "\n")

        if _d7:
            S = torch.cat(_dbg_S, dim=0)
            experts = {"mean": torch.cat(_usr_TERM, 0), "recency": torch.cat(_usr_TERM_rec, 0),
                       "max": torch.cat(_usr_TERM_max, 0)}
            experts["ens"] = (experts["mean"] + experts["recency"] + experts["max"]) / 3.0   # MoE ensemble
            _tgt = target_items.detach().cpu().numpy()
            for name in ["recency", "max", "ens"]:
                TERM = experts[name]
                _rows = [FULL_HDR]
                for g in [0.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
                    adj = S + g * TERM
                    rk = (adj > adj[:, 0:1]).sum(dim=1).detach().cpu().numpy().astype(float)
                    _rows.append(_full_row(g, rk))
                with open(os.path.join(self.args.output_dir, "usr_sem_sweep_%s.csv" % name), "w") as _f:
                    _f.write("\n".join(_rows) + "\n")
            self.logger.info("D7+MoE sweeps written (recency, max, ens)")
            # (A) cold-item stratification: HR of the max-pool D6 term by TARGET-item training frequency
            _pop = np.asarray(self.item_pop); _tp = _pop[np.clip(_tgt.astype(int), 0, len(_pop) - 1)]
            _gg = [0.0, 8.0, 16.0, 32.0]; _Tm = experts["max"]
            _rk = {}
            for g in _gg:
                adj = S + g * _Tm
                _rk[g] = (adj > adj[:, 0:1]).sum(1).detach().cpu().numpy()
            _cr = ["bin(n)," + ",".join("HR@10_g%g" % g for g in _gg)]
            for lo, hi in [(0, 2), (3, 5), (6, 20), (21, 100), (101, 10 ** 9)]:
                _sel = (_tp >= lo) & (_tp <= hi)
                if _sel.sum() == 0:
                    continue
                _cr.append("%d-%d(%d)," % (lo, hi, int(_sel.sum())) +
                           ",".join("%.4f" % (_rk[g][_sel] < 10).mean() for g in _gg))
            with open(os.path.join(self.args.output_dir, "usr_sem_cold.csv"), "w") as _f:
                _f.write("\n".join(_cr) + "\n")

        if _usr and getattr(self.args, "usr_sem_seg", False):
            # per-segment: apply gamma ONLY to short-history (tail) users (seq_len < ts_user)
            short = (seq_len < self.args.ts_user).float().unsqueeze(1)   # (N,1)
            _segTERM = experts["max"] if _d7 else TERM   # route the max-pool (headline) content term to Tail Users
            self.logger.info("Per-segment user-semantic sweep (gamma on Tail-Users / short-history only):")
            _rows = [FULL_HDR.replace("gamma,", "gamma_tail,", 1)]
            for g in [0.0, 8.0, 16.0, 32.0, 64.0]:
                adj = S + (short * g) * _segTERM
                rk = (adj > adj[:, 0:1]).sum(dim=1).detach().cpu().numpy().astype(float)
                _rows.append(_full_row(g, rk))
            with open(os.path.join(self.args.output_dir, "usr_sem_seg_sweep.csv"), "w") as _f:
                _f.write("\n".join(_rows) + "\n")

        if _moe:
            self._moe_analysis(torch.cat(_moe_col, 0), torch.cat(_moe_sem, 0), torch.cat(_moe_max, 0),
                               torch.cat(_moe_cand, 0).long(), seq_len, target_items,
                               torch.cat(_dbg_S, 0) if _dbg_S else None)
        return res_dict



    def _moe_analysis(self, COL, SEM, MX, CAND, seq_len, target_items, S_opt=None):
        """E22-E24: two-specialist routing. Decompose the score into collaborative + semantic
        experts (COL/SEM) plus the max-pool content term (MX), then write: moe_diag.csv (the 2x2
        user x item expert diagnostic), moe_route.csv (per-candidate router alpha_tail sweep),
        moe_route_v2.csv (user-gate / soft / ts_item variants) and moe_route_full.csv (group breakdown)."""
        FULL = COL + SEM
        if S_opt is not None:   # sanity: collab+semantic must reconstruct the model's predict score
            self.logger.info("  [moe] |full - predict| max = %.2e" % (FULL - S_opt).abs().max().item())
        experts = {"collab": COL, "semantic": SEM, "full": FULL, "full+max": FULL + 16.0 * MX}
        enames = ["collab", "semantic", "full", "full+max"]
        hit = {}
        for e in enames:
            rk = (experts[e] > experts[e][:, 0:1]).sum(1).detach().cpu().numpy()
            hit[e] = (rk < 10).astype(float)
        sl = seq_len.detach().cpu().numpy()
        _pop = np.asarray(self.item_pop)
        tp = _pop[np.clip(target_items.long().detach().cpu().numpy(), 0, len(_pop) - 1)]
        tu = sl < self.args.ts_user; ti = tp < self.args.ts_item
        cells = [("TailU-TailI", tu & ti), ("TailU-HeadI", tu & ~ti),
                 ("HeadU-TailI", ~tu & ti), ("HeadU-HeadI", ~tu & ~ti)]
        routed = np.zeros(len(sl))
        rows = ["cell,N," + ",".join(enames) + ",best,best-full"]
        for cname, sel in cells:
            if sel.sum() == 0:
                continue
            hrs = {e: hit[e][sel].mean() for e in enames}
            best = max(hrs, key=hrs.get); routed[sel] = hit[best][sel]
            rows.append("%s,%d,%s,%s,%+.4f" % (cname, int(sel.sum()),
                        ",".join("%.4f" % hrs[e] for e in enames), best, hrs[best] - hrs["full"]))
        single = {e: hit[e].mean() for e in enames}; bs = max(single, key=single.get)
        rows.append("OVERALL,%d,%s,routed=%.4f,vs_best_single(%s)=%.4f[%+.4f]" % (
            len(sl), ",".join("%.4f" % single[e] for e in enames), routed.mean(), bs, single[bs], routed.mean() - single[bs]))
        with open(os.path.join(self.args.output_dir, "moe_diag.csv"), "w") as _f:
            _f.write("\n".join(rows) + "\n")
        self.logger.info("  [moe] routed(oracle per-cell)=%.4f vs best_single %s=%.4f" % (routed.mean(), bs, single[bs]))
        # Stage-2 realizable router: per-CANDIDATE, gate the collaborative score by item popularity.
        # score(u,i) = alpha_i*collab + semantic + 16*max ; alpha_i = 1 (head item) or alpha_tail (tail item).
        # alpha_tail=1 == full+max (baseline); alpha_tail=0 == semantic+content (drop collab) for cold items.
        cp = torch.as_tensor(_pop, device=COL.device, dtype=torch.float32)[CAND.clamp(0, len(_pop) - 1)]
        is_head = (cp >= self.args.ts_item).float()                      # (Ntot, n_cand)
        rrows = ["alpha_tail,HR@10,NDCG@10,coldcold_HR,tailitem_HR,headitem_HR"]
        for at in [1.0, 0.5, 0.25, 0.0]:
            alpha = is_head + (1.0 - is_head) * at
            sc = alpha * COL + SEM + 16.0 * MX
            rk = (sc > sc[:, 0:1]).sum(1).detach().cpu().numpy()
            h = (rk < 10).astype(float); nd = (1.0 / np.log2(rk + 2.0)) * (rk < 10)
            rrows.append("%.2f,%.5f,%.5f,%.5f,%.5f,%.5f" % (at, h.mean(), nd.mean(),
                         h[tu & ti].mean(), h[ti].mean(), h[~ti].mean()))
        with open(os.path.join(self.args.output_dir, "moe_route.csv"), "w") as _f:
            _f.write("\n".join(rrows) + "\n")
        self.logger.info("  [moe] realizable router written (alpha_tail sweep)")
        # router VARIANTS: user-level gate, soft item gates, ts_item sweep (all vs full+max baseline)
        def _rowfor(name, sc):
            rk = (sc > sc[:, 0:1]).sum(1).detach().cpu().numpy()
            h = (rk < 10).astype(float); nd = (1.0 / np.log2(rk + 2.0)) * (rk < 10)
            return "%s,%.5f,%.5f,%.5f,%.5f,%.5f" % (name, h.mean(), nd.mean(),
                   h[tu & ti].mean(), h[ti].mean(), h[~ti].mean())
        uhead = torch.as_tensor(sl >= self.args.ts_user, device=COL.device, dtype=torch.float32).unsqueeze(1)
        T = float(self.args.ts_item)
        def gate_item(thr, at):
            ih = (cp >= thr).float(); return ih + (1.0 - ih) * at
        v2 = ["config,HR@10,NDCG@10,coldcold_HR,tailitem_HR,headitem_HR"]
        v2.append(_rowfor("baseline_fullmax", COL + SEM + 16.0 * MX))
        v2.append(_rowfor("item_hard_a0", gate_item(T, 0.0) * COL + SEM + 16.0 * MX))
        v2.append(_rowfor("item_soft_lin", (cp / T).clamp(max=1.0) * COL + SEM + 16.0 * MX))
        v2.append(_rowfor("item_soft_log", (torch.log1p(cp) / float(np.log1p(T))).clamp(max=1.0) * COL + SEM + 16.0 * MX))
        v2.append(_rowfor("user_gate_a0", uhead * COL + SEM + 16.0 * MX))
        v2.append(_rowfor("user_gate_a50", (uhead + (1.0 - uhead) * 0.5) * COL + SEM + 16.0 * MX))
        for k in [0.5, 2.0, 4.0]:
            v2.append(_rowfor("ts_x%.1f_a0" % k, gate_item(T * k, 0.0) * COL + SEM + 16.0 * MX))
        with open(os.path.join(self.args.output_dir, "moe_route_v2.csv"), "w") as _f:
            _f.write("\n".join(v2) + "\n")
        self.logger.info("  [moe] router variants written (user-gate / soft / ts-sweep)")
        # D6/D7-style full group breakdown: full+max (base) vs router (alpha_tail=0, drop collab on cold items)
        _tgt_np = target_items.long().detach().cpu().numpy()
        def _grp(sc):
            rk = (sc > sc[:, 0:1]).sum(1).detach().cpu().numpy().astype(float)
            r = metric_report(rk); rp = metric_pop_report(rk, self.item_pop, _tgt_np, args=self.args)
            rl = metric_len_report(rk, sl, aug_len=self.args.aug_seq_len, args=self.args)
            g = lambda d, k: d.get(k, float("nan"))
            return [r["HR@10"], r["NDCG@10"], g(rp, "Tail HR@10"), g(rp, "Tail NDCG@10"),
                    g(rp, "Popular HR@10"), g(rp, "Popular NDCG@10"), g(rl, "Short HR@10"),
                    g(rl, "Short NDCG@10"), g(rl, "Long HR@10"), g(rl, "Long NDCG@10")]
        fb = ["config,HR@10,NDCG@10,TailItem_HR,TailItem_N,HeadItem_HR,HeadItem_N,TailUser_HR,TailUser_N,HeadUser_HR,HeadUser_N"]
        fb.append("full+max," + ",".join("%.5f" % x for x in _grp(COL + SEM + 16.0 * MX)))
        fb.append("router_a0," + ",".join("%.5f" % x for x in _grp(gate_item(T, 0.0) * COL + SEM + 16.0 * MX)))
        with open(os.path.join(self.args.output_dir, "moe_route_full.csv"), "w") as _f:
            _f.write("\n".join(fb) + "\n")

    def _eval_fullrank(self, test_loader, desc):
        """(D) Full-catalog ranking: rank the target against ALL items (not 100 sampled negatives),
        with the max-pool D6 term swept over gamma. Writes fullrank_sweep.csv. SASRec only."""
        dev = self.device; N = self.item_num
        all_items = torch.arange(1, N + 1, device=dev)
        with torch.no_grad():
            all_emb = self.model._get_embedding(all_items)      # (N, 2H)
            all_llm = self.model.llm_item_emb(all_items)        # (N, d_llm) frozen
        GAMMAS = [0.0, 8.0, 16.0, 32.0]
        ranks = {g: [] for g in GAMMAS}
        for batch in tqdm(test_loader, desc=desc):
            batch = tuple(t.to(dev) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq = inputs["seq"]; pos = inputs["pos"].long()
            with torch.no_grad():
                feat = self.model.log2feats(seq, inputs["positions"])[:, -1, :]   # (B, 2H)
                base = feat @ all_emb.t()                                          # (B, N)
                # mask already-seen items (standard full-ranking practice): otherwise the max-pool term
                # gives every history item self-similarity ~1 and floats consumed items to the top.
                _nz = torch.nonzero(seq > 0, as_tuple=False)
                _seen = torch.zeros_like(base, dtype=torch.bool)
                _seen[_nz[:, 0], (seq[seq > 0] - 1).clamp(0, N - 1)] = True
                base = base.masked_fill(_seen, float("-inf"))
                eh = self.model.llm_item_emb(seq); m = (seq > 0)
                term = torch.empty_like(base)
                for st in range(0, N, 2000):
                    ch = all_llm[st:st + 2000]
                    sims = torch.einsum("bld,cd->blc", eh, ch).masked_fill(~m.unsqueeze(-1), float("-inf"))
                    term[:, st:st + 2000] = torch.nan_to_num(sims.max(1).values, neginf=0.0)  # empty-history -> 0
                tgt = (pos - 1).clamp(0, N - 1)
                for g in GAMMAS:
                    adj = base + g * term
                    ts = adj.gather(1, tgt.unsqueeze(1))
                    ranks[g].append((adj > ts).sum(1).detach().cpu().numpy())
        rows = ["gamma,HR@10,NDCG@10,MRR"]
        for g in GAMMAS:
            rk = np.concatenate(ranks[g]).astype(float)
            hr = float((rk < 10).mean())
            ndcg = float(np.where(rk < 10, 1.0 / np.log2(rk + 2), 0.0).mean())
            mrr = float((1.0 / (rk + 1)).mean())
            rows.append("%.1f,%.5f,%.5f,%.5f" % (g, hr, ndcg, mrr))
            self.logger.info("  full-rank gamma=%.0f: HR@10 %.4f NDCG %.4f MRR %.4f" % (g, hr, ndcg, mrr))
        with open(os.path.join(self.args.output_dir, "fullrank_sweep.csv"), "w") as f:
            f.write("\n".join(rows) + "\n")
        return {"HR@10": float((np.concatenate(ranks[0.0]) < 10).mean())}

    def _final_feat(self, inputs):
        """Backbone-agnostic user representation at the prediction position (=[collab|semantic], 2H),
        replicating each model's predict()."""
        m = getattr(self.args, "model_name", "")
        seq = inputs["seq"]
        if m == "llmesr_gru4rec":
            return self.model.log2feats(seq)[:, -1, :]
        if m == "llmesr_bert4rec":
            pos = inputs["positions"]
            log_seqs = torch.cat([seq, self.model.mask_token * torch.ones(seq.shape[0], 1, device=seq.device)], dim=1)
            pos2 = torch.cat([pos, (pos[:, -1] + 1).unsqueeze(1)], dim=1)
            return self.model.log2feats(log_seqs[:, 1:].long(), pos2[:, 1:].long())[:, -1, :]
        return self.model.log2feats(seq, inputs["positions"])[:, -1, :]

    def save_user_emb(self):

        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        try:
            self.model.load_state_dict(model_state_dict['state_dict'])
        except:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        test_loader = self.test_loader

        self.model.eval()
        user_emb = torch.empty(0).to(self.device)
        desc = 'Running'

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            
            with torch.no_grad():

                per_user_emb = self.model.get_user_emb(**inputs)
                user_emb = torch.cat([user_emb, per_user_emb], dim=0)
        
        user_emb = user_emb.detach().cpu().numpy()
        import pickle
        pickle.dump(user_emb, open("./usr_emb_sasrec.pkl", "wb"))


    
    def test_group(self):

        print('')
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running Group test **********")
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)
        test_loader = self.test_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        # res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        # res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)
        hr_len, ndcg_len, count_len = metric_len_5group(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), [5, 10, 15, 20])
        hr_pop, ndcg_pop, count_pop = metric_pop_5group(pred_rank.detach().cpu().numpy(), self.item_pop,  target_items.detach().cpu().numpy(), [10, 30, 60, 100])

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("User Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_len, ndcg_len)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        self.logger.info("Item Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_pop, ndcg_pop)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        
        
        return res_dict
    


class CL4SRecTrainer(SeqTrainer):

    def __init__(self, args, logger, writer, device, generator):
        
        super().__init__(args, logger, writer, device, generator)


    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq, pos, neg, positions, aug1, aug2 = batch
            seq, pos, neg, positions, aug1, aug2 = seq.long(), pos.long(), neg.long(), positions.long(), aug1.long(), aug2.long()
            aug = (aug1, aug2)
            loss = self.model(seq, pos, neg, positions, aug)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



class SSEPTTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            loss = self.model(seq_user, pos_user, neg_user, seq, pos, neg, positions)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            try:
                self.model.load_state_dict(model_state_dict['state_dict'])
            except:
                self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            seq_len = torch.cat([seq_len, torch.sum(seq>0, dim=1)])

            with torch.no_grad():

                pred_logits = -self.model.predict(seq_user, seq, torch.cat([pos_user.unsqueeze(1), neg_user], dim=1), torch.cat([pos.unsqueeze(1), neg], dim=1), positions)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len)
        
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        for k, v in res_len_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict}

        if test:
            record_csv(self.args, res_dict)

            # per-user scores for downstream paired t-tests, written to
            # <output_dir>/per_user_scores/{ndcg,hr}_s{seed}.npy, shape [N_users] float64
            per_ndcg, per_hr = per_user_ndcg_hr(pred_rank.detach().cpu().numpy())
            score_dir = os.path.join(self.args.output_dir, "per_user_scores")
            os.makedirs(score_dir, exist_ok=True)
            np.save(os.path.join(score_dir, f"ndcg_s{self.args.seed}.npy"), per_ndcg)
            np.save(os.path.join(score_dir, f"hr_s{self.args.seed}.npy"),   per_hr)
            self.logger.info(f"Per-user scores saved to {score_dir}")

        return res_dict
