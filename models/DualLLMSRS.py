"""Dual-view sequential backbones (SASRec / GRU4Rec / Bert4Rec).

Each item is represented in two views: a collaborative (PCA-initialised id) embedding and a frozen LLM
semantic embedding passed through an adapter, concatenated so the final score splits exactly as
``collab_score + semantic_score``. Architecture/initialisation extensions are flag-gated here:
``_build_adapter`` (E1-E4 activation/width/depth/residual), the collaborative-side adapter (E5),
LoRA semi-unfreeze (E7), popularity-gated fusion (E8), and the encoder swap loaders
``load_llm_item_emb``/``load_id_item_emb`` (E25). Defaults reproduce the original LLM-ESR.
"""
# here put the import lib
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.GRU4Rec import GRU4Rec
from models.SASRec import SASRec_seq, SASRecBackbone
from models.Bert4Rec import Bert4Rec
from models.utils import Multi_CrossAttention


def load_llm_item_emb(args):
    """Load the frozen LLM item embedding (the semantic view), with encoder swap (E25).
    Reads from --emb_path when set (a directory containing item.npy, or a direct .npy/.pkl file),
    so the embedding ablation only swaps which encoder's vectors are loaded; otherwise falls back to
    data/<dataset>/handled/itm_emb_np.pkl. With --d_llm>0 the loaded dim is asserted to match."""
    emb_path = getattr(args, "emb_path", "")
    if emb_path:
        path = os.path.join(emb_path, "item.npy") if os.path.isdir(emb_path) else emb_path
        llm_item_emb = np.load(path) if path.endswith(".npy") else pickle.load(open(path, "rb"))
    else:
        path = os.path.join("data/" + args.dataset + "/handled/", "itm_emb_np.pkl")
        llm_item_emb = pickle.load(open(path, "rb"))
    llm_item_emb = np.asarray(llm_item_emb, dtype=np.float32)
    d_llm = getattr(args, "d_llm", 0)
    if d_llm and llm_item_emb.shape[1] != d_llm:
        raise ValueError("Loaded LLM item embedding dim %d from %s does not match --d_llm %d"
                         % (llm_item_emb.shape[1], path, d_llm))
    return llm_item_emb


def load_id_item_emb(args):
    """Load the PCA-reduced item embedding that initialises the collaborative (id) view. For the
    embedding ablation the PCA is recomputed per run, so when --emb_path is a directory containing a
    run-local pca64.npy we load that; otherwise fall back to data/<dataset>/handled/pca64_itm_emb_np.pkl."""
    emb_path = getattr(args, "emb_path", "")
    if emb_path and os.path.isdir(emb_path):
        cand = os.path.join(emb_path, "pca64.npy")
        if os.path.exists(cand):
            return np.asarray(np.load(cand), dtype=np.float32)
    path = os.path.join("data/" + args.dataset + "/handled/", "pca64_itm_emb_np.pkl")
    return np.asarray(pickle.load(open(path, "rb")), dtype=np.float32)


class ResidualAdapter(nn.Module):
    """Adapter branch plus a learned linear skip: output = adapter(x) + skip(x).

    in_dim != out_dim so there is no identity shortcut; skip is a bias-free
    Linear(in_dim, out_dim) and the adapter learns a residual on top of it.
    For linear adapters the sum stays linear, but the factored form (two
    low-rank matrices) can reach full rank faster than a single matrix.
    """

    def __init__(self, adapter: nn.Module, in_dim: int, out_dim: int):
        super().__init__()
        self.adapter = adapter
        self.skip    = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x) + self.skip(x)


def _build_adapter(in_dim: int,
                   out_dim: int,
                   act: str = "none",
                   num_layers: int = 2,
                   hidden_dim: int = -1,
                   residual: bool = False) -> nn.Module:
    """Build the LLM-embedding -> hidden_size adapter MLP.

    in_dim = LLM embedding dim (e.g. 1536 for ada-002); out_dim = hidden_size (e.g. 64).
    act = activation between layers (none|relu|gelu|silu|tanh); "none" is pure linear
    (default, matches the original). num_layers: 1 = single Linear, 2 = in->H->out
    (original), 3 = in->H->H//2->out, with H = hidden_dim if hidden_dim > 0 else in_dim//2.
    residual = if True, wrap the MLP with a linear skip.
    """

    _act_map = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
    }

    def _get_act():
        if act == "none":
            return []
        if act not in _act_map:
            raise ValueError(f"Unknown adapter_act '{act}'. "
                             f"Choose from: none, relu, gelu, silu, tanh")
        return [_act_map[act]]

    if num_layers == 1:
        # residual on a single linear is degenerate (skip + linear = linear)
        # so we ignore the residual flag here and just return the linear map
        return nn.Linear(in_dim, out_dim)

    mid = hidden_dim if hidden_dim > 0 else in_dim // 2

    # generic N-layer (>=2) MLP: in -> mid -> mid/2 -> ... -> out, halving the hidden width each
    # step but never below out_dim. Reproduces the original 2- and 3-layer shapes exactly.
    dims = [in_dim]
    h = mid
    for _ in range(num_layers - 1):
        dims.append(max(h, out_dim)); h = h // 2
    dims.append(out_dim)
    layers = []
    for k in range(num_layers):
        layers.append(nn.Linear(dims[k], dims[k + 1]))
        if k < num_layers - 1:
            layers += _get_act()
    mlp = nn.Sequential(*layers)

    if residual:
        return ResidualAdapter(mlp, in_dim, out_dim)
    return mlp



class DualLLMGRU4Rec(GRU4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att
        self.hidden_size = args.hidden_size   # used by log2feats QKV/sqrt(D) scaling

        # load llm embedding as item embedding
        llm_item_emb = load_llm_item_emb(args)
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default

        self.adapter = _build_adapter(
            in_dim=llm_item_emb.shape[1],
            out_dim=args.hidden_size,
            act=getattr(args, "adapter_act", "none"),
            num_layers=getattr(args, "adapter_layers", 2),
            hidden_dim=getattr(args, "adapter_hidden_dim", -1),
            residual=getattr(args, "adapter_residual", False),
        )

        id_item_emb = load_id_item_emb(args)
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)

        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()


    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs):

        id_seqs = self.id_item_emb(log_seqs)
        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats



class DualLLMSASRec(SASRec_seq):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att
        # ablation switches (Table 2); all default to False / original behavior
        self.co_view = getattr(args, "co_view", False)        # ONLY collaborative view = w/o Se-view
        self.se_view = getattr(args, "se_view", False)        # ONLY semantic view      = w/o Co-view
        self.split_backbone = getattr(args, "split_backbone", False)   # w/o Share

        # load llm embedding as item embedding
        llm_item_emb = load_llm_item_emb(args)
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default

        # extension: LoRA on the frozen LLM embedding (semi-unfreeze). e_eff = e_frozen + A[item]@B,
        # B init 0 => delta 0 => identical to baseline at init. A,B trainable; frozen emb stays frozen.
        self.llm_lora_rank = getattr(args, "llm_lora_rank", 0)
        if self.llm_lora_rank > 0:
            self.lora_A = nn.Embedding(self.item_num + 2, self.llm_lora_rank)
            self.lora_B = nn.Parameter(torch.zeros(self.llm_lora_rank, llm_item_emb.shape[1]))
            nn.init.normal_(self.lora_A.weight, std=0.02)

        # D6 train-time: learnable-gamma user-semantic scoring term (added in forward+predict).
        # gamma init 0 => identical to baseline at init, then learned. (eval-only D6 uses --usr_sem instead)
        self.usr_sem_train = getattr(args, "usr_sem_train", False)
        if self.usr_sem_train:
            self.usr_gamma = nn.Parameter(torch.full((1,), float(getattr(args, "usr_sem_train_init", 0.0))))

        # adapter_act       : "none" | "relu" | "gelu" | "silu" | "tanh"  (default: "none" = pure linear)
        # adapter_layers    : 1 | 2 | 3  (default: 2, matching the original released code)
        # adapter_hidden_dim: intermediate dim; -1 = in_dim // 2 (default)
        self.adapter = _build_adapter(
            in_dim=llm_item_emb.shape[1],
            out_dim=args.hidden_size,
            act=getattr(args, "adapter_act", "none"),
            num_layers=getattr(args, "adapter_layers", 2),
            hidden_dim=getattr(args, "adapter_hidden_dim", -1),
            residual=getattr(args, "adapter_residual", False),
        )

        # extension: optional collaborative-side adapter. co_adapter_dim>0 => the collaborative
        # embedding has dim d_co (init from pca{d_co}) and a linear adapter maps d_co -> hidden_size.
        # co_adapter_dim=-1 (default) reproduces the original (64-d PCA embedding, no adapter).
        self.hidden_size = args.hidden_size
        self.co_adapter_dim = getattr(args, "co_adapter_dim", -1)
        self.co_adapter = None
        if getattr(args, "id_random_init", False):      # ablation: Random Init (no PCA LLM emb)
            self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        elif self.co_adapter_dim > 0:
            _pca = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca%d_itm_emb_np.pkl" % self.co_adapter_dim), "rb"))
            _pca = np.insert(_pca, 0, values=np.zeros((1, _pca.shape[1])), axis=0)
            _pca = np.concatenate([_pca, np.zeros((1, _pca.shape[1]))], axis=0)
            self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(_pca))
            self.id_item_emb.weight.requires_grad = True
            self.co_adapter = nn.Linear(self.co_adapter_dim, args.hidden_size)
        else:
            # extension: swap PCA for a learnable dimensionality-reduction init of the collaborative
            # embedding. --id_init_path points to an (item_num x hidden) .pkl; empty = original pca64.
            _init_path = getattr(args, "id_init_path", "")
            if _init_path:
                id_item_emb = pickle.load(open(_init_path, "rb"))
            else:
                id_item_emb = load_id_item_emb(args)   # --emb_path run-local pca64 if set, else default pca64
            id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
            id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
            self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))
            self.id_item_emb.weight.requires_grad = True   # the grad is false in default

        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if self.split_backbone:     # ablation: w/o Share -> separate encoder for the semantic view
            self.backbone_se = SASRecBackbone(device, args)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        # extension: popularity-gated dual-view fusion (item-side gate; init g=0.5 == baseline)
        self.pop_gate = getattr(args, "pop_gate", "none")
        if self.pop_gate != "none":
            _pop = np.zeros(self.item_num + 2, dtype=np.float64)
            with open(os.path.join("data/"+args.dataset+"/handled/", "inter.txt")) as _f:
                for _line in _f:
                    _p = _line.split()
                    if len(_p) >= 2:
                        _j = int(_p[1])
                        if 0 <= _j < _pop.shape[0]:
                            _pop[_j] += 1.0
            _lp = np.log1p(_pop)
            _real = _lp[1:self.item_num + 1]
            _z = (_lp - _real.mean()) / (_real.std() + 1e-8)
            _z[0] = 0.0; _z[self.item_num + 1] = 0.0    # padding / mask -> neutral
            self.register_buffer("item_logpop", torch.tensor(_z, dtype=torch.float32))
            if self.pop_gate == "pop":          # 2-param parametric gate g=sigmoid(w*z+b)
                self.gate_w = nn.Parameter(torch.zeros(1))
                self.gate_b = nn.Parameter(torch.zeros(1))
            elif self.pop_gate == "item":       # per-item learnable gate (init 0 -> g=0.5)
                self.gate_item = nn.Embedding(self.item_num + 2, 1)
                nn.init.zeros_(self.gate_item.weight)

        if getattr(args, "id_random_init", False):
            # random-init id embedding should be initialized like any other module (xavier)
            _filt = ["llm_item_emb"]
        else:
            _filt = ["llm_item_emb", "id_item_emb"]
        if self.pop_gate == "item":
            _filt.append("gate_item")           # keep per-item gate at its zero init
        if self.llm_lora_rank > 0:
            _filt += ["lora_A", "lora_B"]       # keep lora_B at zero init (delta = 0 => baseline at init)
        self.filter_init_modules = _filt
        self._init_weights()


    def _id_emb(self, log_seqs):
        e = self.id_item_emb(log_seqs)
        if self.co_adapter is not None:        # collaborative adapter: d_co -> hidden_size
            e = self.co_adapter(e)
        return e

    def _llm_emb(self, log_seqs):
        e = self.llm_item_emb(log_seqs)
        if self.llm_lora_rank > 0:             # semi-unfreeze: frozen emb + trainable low-rank delta
            e = e + self.lora_A(log_seqs) @ self.lora_B
        return e

    def _usr_sem_score(self, seq, item_ids, causal):
        """D6 user-semantic term: <user history LLM-centroid, candidate LLM-emb>.
        causal=False (inference): full-history centroid (B,d) vs candidates (B,n) -> (B,n).
        causal=True  (training):  cumulative centroid up to each position (no future leak),
        scored against that position's target item_ids (B,L) -> (B,L)."""
        le = self.llm_item_emb                         # frozen LLM item embedding
        e_hist = le(seq)                               # (B,L,d)
        m = (seq > 0).float().unsqueeze(-1)            # (B,L,1)
        if causal:
            csum = torch.cumsum(e_hist * m, dim=1)
            ccnt = torch.cumsum(m, dim=1).clamp(min=1.0)
            c = csum / ccnt                            # (B,L,d) causal centroid at each step
            return (c * le(item_ids)).sum(-1)          # (B,L)
        c = (e_hist * m).sum(1) / m.sum(1).clamp(min=1.0)   # (B,d) full-history centroid
        return (le(item_ids) * c.unsqueeze(1)).sum(-1)      # (B,n)

    def _get_embedding(self, log_seqs):

        id_seq_emb = self._id_emb(log_seqs)
        llm_seq_emb = self._llm_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        if self.co_view:        # w/o Se-view: collaborative embedding only
            return id_seq_emb
        if self.se_view:        # w/o Co-view: semantic embedding only
            return llm_seq_emb

        if self.pop_gate != "none":   # popularity gate: tail items lean semantic, head lean collaborative
            _z = self.item_logpop[log_seqs].unsqueeze(-1)
            if self.pop_gate == "pop":
                _g = torch.sigmoid(self.gate_w * _z + self.gate_b)
            else:
                _g = torch.sigmoid(self.gate_item(log_seqs))
            id_seq_emb = id_seq_emb * (2.0 * _g)            # collaborative weighted by g
            llm_seq_emb = llm_seq_emb * (2.0 * (1.0 - _g))  # semantic weighted by (1-g)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):

        if self.co_view or self.se_view:    # single-view ablations: one branch, one encoder
            if self.co_view:
                seqs = self._id_emb(log_seqs)
            else:
                seqs = self.adapter(self._llm_emb(log_seqs))
            seqs *= self.hidden_size ** 0.5  # QKV/sqrt(D) (hidden_size == id emb dim unless co_adapter is used)
            seqs += self.pos_emb(positions.long())
            seqs = self.emb_dropout(seqs)
            return self.backbone(seqs, log_seqs)

        id_seqs = self._id_emb(log_seqs)
        id_seqs *= self.hidden_size ** 0.5  # QKV/sqrt(D) (hidden_size == id emb dim unless co_adapter is used)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self._llm_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.hidden_size ** 0.5  # QKV/sqrt(D) (hidden_size == id emb dim unless co_adapter is used)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
            cross_id_seqs = 1 * cross_id_seqs + 0 * id_seqs
            cross_llm_seqs = 1 * cross_llm_seqs + 0 * llm_seqs
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        se_backbone = self.backbone_se if self.split_backbone else self.backbone
        llm_log_feats = se_backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats



class DualLLMBert4Rec(Bert4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att
        self.hidden_size = args.hidden_size   # used by log2feats QKV/sqrt(D) scaling

        # load llm embedding as item embedding
        llm_item_emb = load_llm_item_emb(args)
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default

        self.adapter = _build_adapter(
            in_dim=llm_item_emb.shape[1],
            out_dim=args.hidden_size,
            act=getattr(args, "adapter_act", "none"),
            num_layers=getattr(args, "adapter_layers", 2),
            hidden_dim=getattr(args, "adapter_hidden_dim", -1),
            residual=getattr(args, "adapter_residual", False),
        )

        id_item_emb = load_id_item_emb(args)
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)

        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()


    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):

        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= self.hidden_size ** 0.5  # QKV/sqrt(D) (hidden_size == id emb dim unless co_adapter is used)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.hidden_size ** 0.5  # QKV/sqrt(D) (hidden_size == id emb dim unless co_adapter is used)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
