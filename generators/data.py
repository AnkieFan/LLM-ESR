# here put the import lib
import os
import copy
import random
import numpy as np
from torch.utils.data import Dataset
from utils.utils import random_neq
import pickle


def sample_sim(mode, pool, k):
    """Self-distillation teacher sampling. 'topn' = first-k (original); 'stochastic' (E11/E12) =
    k random teachers drawn from the top-50 semantic-neighbour pool."""
    if mode == "stochastic":
        top = pool[:50]
        chosen = np.random.choice(top, size=min(k, len(top)), replace=False)
        return list(chosen)
    return pool[:k]


class SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation'''

    def __init__(self, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.var_name = ["seq", "pos", "neg", "positions"]
        self.hard_neg_ratio = 0.0   # overridden by SeqDatasetAllUser when --hard_neg_ratio>0
        self.hard_pool = None

    def _hard_neg(self, pos, non_neg):
        # draw a negative from the K most LLM-similar items to the positive, excluding padding(0)
        # and anything already in the sequence / already-picked negatives; fall back to random.
        valid = [int(c) for c in self.hard_pool[pos] if c != 0 and c not in non_neg]
        if valid:
            return random.choice(valid)
        return random_neq(1, self.item_num+1, non_neg)


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions
    


class SeqDatasetAllUser(SeqDataset):
    '''The train dataset for Sequential recommendation'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__(data, item_num, max_len, neg_num)
        self.sim_user_num = args.sim_user_num
        self.sampling_method = getattr(args, "sampling_method", "topn")
        self.sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", getattr(args, "sim_user_file", "sim_user_100.pkl")), "rb"))
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        ### get the sequence of similar user (map augmented index -> original user for sim_users)
        _oi = self.user_map[index] if getattr(self, "user_map", None) is not None else index
        sim_users = sample_sim(self.sampling_method, self.sim_users[_oi], self.sim_user_num)
        sim_seq, sim_positions = [], []
        for sim_user in sim_users:
            meta_seq, meta_positions = self._get_user_seq(sim_user)
            sim_seq.append(meta_seq)
            sim_positions.append(meta_positions)
        
        sim_seq = np.array(sim_seq)
        sim_positions = np.array(sim_positions)

        return seq, pos, neg, positions, index, sim_seq, sim_positions
    

    def _get_user_seq(self, user):

        ### get the sequence of required user (from ORIGINAL seqs, not the augmented prefix list)
        inter = (self.orig_seqs if getattr(self, "orig_seqs", None) is not None else self.data)[user]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions



class Seq2SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation with seq-to-seq loss'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.aug_seq = args.aug_seq
        self.aug_seq_len = args.aug_seq_len
        self.var_name = ["seq", "pos", "neg", "positions"]
        # extension: hard-negative sampling. with prob hard_neg_ratio, draw the per-position training
        # negative from the K most LLM-similar items to that position's positive target (precomputed
        # pool) instead of uniform random. hard_neg_ratio=0 (default) => original random behavior.
        self.hard_neg_ratio = getattr(args, "hard_neg_ratio", 0.0)
        self.hard_pool = None
        if self.hard_neg_ratio > 0:
            _k = getattr(args, "hard_neg_k", 50)
            self.hard_pool = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", "hard_neg_k%d_pool.pkl" % _k), "rb"))

    def _hard_neg(self, target, non_neg):
        valid = [int(c) for c in self.hard_pool[target] if c != 0 and c not in non_neg]
        if valid:
            return random.choice(valid)
        return random_neq(1, self.item_num+1, non_neg)


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if self.hard_pool is not None and random.random() < self.hard_neg_ratio:
                neg[idx] = self._hard_neg(nxt, non_neg)
            else:
                neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if self.aug_seq:
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions



class Seq2SeqDatasetAllUser(Seq2SeqDataset):

    def __init__(self, args, data, item_num, max_len, neg_num=1, orig_seqs=None, user_map=None):

        super().__init__(args, data, item_num, max_len, neg_num)
        self.sim_user_num = args.sim_user_num
        self.sampling_method = getattr(args, "sampling_method", "topn")
        self.sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", getattr(args, "sim_user_file", "sim_user_100.pkl")), "rb"))
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions"]
        # augmentation: orig_seqs = original (un-augmented) sequences used for sim-user lookup;
        # user_map[index] = original 0-based user index of augmented example `index` (for sim_users).
        # both None => no augmentation (original behaviour: index is the user index, data is orig_seqs).
        self.orig_seqs = orig_seqs if orig_seqs is not None else data
        self.user_map = user_map


    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if self.hard_pool is not None and random.random() < self.hard_neg_ratio:
                neg[idx] = self._hard_neg(nxt, non_neg)
            else:
                neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if self.aug_seq:
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        ### get the sequence of similar user (map augmented index -> original user for sim_users)
        _oi = self.user_map[index] if getattr(self, "user_map", None) is not None else index
        sim_users = sample_sim(self.sampling_method, self.sim_users[_oi], self.sim_user_num)
        sim_seq, sim_positions = [], []
        for sim_user in sim_users:
            meta_seq, meta_positions = self._get_user_seq(sim_user)
            sim_seq.append(meta_seq)
            sim_positions.append(meta_positions)
        
        sim_seq = np.array(sim_seq)
        sim_positions = np.array(sim_positions)

        return seq, pos, neg, positions, index, sim_seq, sim_positions
    

    def _get_user_seq(self, user):

        ### get the sequence of required user (from ORIGINAL seqs, not the augmented prefix list)
        inter = (self.orig_seqs if getattr(self, "orig_seqs", None) is not None else self.data)[user]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions
    


class BertRecTrainDatasetAllUser(Dataset):
    '''The train dataset for Bert4Rec'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.mask_prob = args.mask_prob
        self.sim_user_num = args.sim_user_num
        self.mask_token = item_num + 1
        self.sampling_method = getattr(args, "sampling_method", "topn")
        self.sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", getattr(args, "sim_user_file", "sim_user_100.pkl")), "rb"))
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions"]


    def __len__(self):

        return 2 * len(self.data)

    def __getitem__(self, index):

        tokens = []
        labels, neg_labels = [], []

        if index >= len(self.data):
            seq = self.data[index - len(self.data)]
            for s in seq:
                tokens.append(s)
                labels.append(0)
                neg_labels.append(0)
            labels[-1] = tokens[-1]
            neg_labels[-1] = random_neq(1, self.item_num+1, seq)
            tokens[-1] = self.mask_token

        else:
            seq = self.data[index]
   
            for s in seq:
                prob = random.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(random.randint(1, self.item_num))
                    else:
                        tokens.append(s)

                    labels.append(s)
                    neg = random_neq(1, self.item_num+1, seq)
                    neg_labels.append(neg)

                else:
                    tokens.append(s)
                    labels.append(0)
                    neg_labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        neg_labels = neg_labels[-self.max_len:]
        pos = list(range(1, len(tokens)+1))
        pos= pos[-self.max_len:]

        mask_len = self.max_len - len(tokens)
        
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        neg_labels = [0] * mask_len + neg_labels
        pos = [0] * mask_len + pos

        if index >= len(self.data):
            user_id = index - len(self.data)
        else:
            user_id = index

        ### get the sequence of similar user
        sim_users = sample_sim(self.sampling_method, self.sim_users[user_id], self.sim_user_num)
        sim_seq, sim_positions = [], []
        for sim_user in sim_users:
            meta_seq, meta_positions = self._get_user_seq(sim_user)
            sim_seq.append(meta_seq)
            sim_positions.append(meta_positions)
        
        sim_seq = np.array(sim_seq)
        sim_positions = np.array(sim_positions)

        return np.array(tokens), np.array(labels), np.array(neg_labels), np.array(pos), user_id, sim_seq, sim_positions


    def _get_user_seq(self, user):

        ### get the sequence of required user (from ORIGINAL seqs, not the augmented prefix list)
        inter = (self.orig_seqs if getattr(self, "orig_seqs", None) is not None else self.data)[user]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions



class Seq2SeqDatasetAllUserFast(Seq2SeqDatasetAllUser):
    '''
    Drop-in faster version of Seq2SeqDatasetAllUser.

    The original rebuilds, for EVERY sample, the input/target/position arrays for
    the user AND for its `sim_user_num` similar users via Python loops
    (`_get_user_seq` called sim_user_num times per __getitem__). That work is
    deterministic and identical across epochs, so we precompute it once for all
    users and reduce __getitem__ to array indexing.

    Negative sampling (random_neq) is left EXACTLY as in the original -- same
    number of RNG draws, same order -- so given the same RNG state the produced
    tensors are bit-identical to Seq2SeqDatasetAllUser.
    '''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        super().__init__(args, data, item_num, max_len, neg_num)
        self._precompute_user_arrays()

    def _build_seq_target_pos(self, inter):
        ml = self.max_len
        seq = np.zeros([ml], dtype=np.int32)
        target = np.zeros([ml], dtype=np.int32)
        nxt = inter[-1]
        idx = ml - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            target[idx] = nxt
            nxt = i
            idx -= 1
            if idx == -1:
                break
        if len(inter) > ml:
            mask_len = 0
            positions = list(range(1, ml + 1))
        else:
            mask_len = ml - (len(inter) - 1)
            positions = list(range(1, len(inter) - 1 + 1))
        positions = positions[-ml:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)
        return seq, target, positions

    def _precompute_user_arrays(self):
        n = len(self.data)
        ml = self.max_len
        self.all_seq = np.zeros((n, ml), dtype=np.int32)
        self.all_target = np.zeros((n, ml), dtype=np.int32)
        self.all_positions = np.zeros((n, ml), dtype=np.int64)
        for u in range(n):
            seq, target, positions = self._build_seq_target_pos(self.data[u])
            self.all_seq[u] = seq
            self.all_target[u] = target
            self.all_positions[u] = positions

    def __getitem__(self, index):
        inter = self.data[index]
        # --- negative sampling: identical RNG behavior to the original ---
        non_neg = copy.deepcopy(inter)
        neg = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            neg[idx] = random_neq(1, self.item_num + 1, non_neg)
            idx -= 1
            if idx == -1:
                break

        seq = self.all_seq[index]
        pos = self.all_target[index]
        positions = self.all_positions[index]

        if self.aug_seq:                       # off by default; preserve behavior
            pos = pos.copy()
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0

        sim_idx = self.sim_users[index][:self.sim_user_num]
        sim_seq = self.all_seq[sim_idx]
        sim_positions = self.all_positions[sim_idx]

        return seq, pos, neg, positions, index, sim_seq, sim_positions
