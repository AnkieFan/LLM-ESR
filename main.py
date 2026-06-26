# here put the import lib
import os
import argparse
import torch

from generators.generator import Seq2SeqGeneratorAllUser
from generators.generator import GeneratorAllUser
from generators.bert_generator import BertGeneratorAllUser
from trainers.sequence_trainer import SeqTrainer
from utils.utils import set_seed
from utils.logger import Logger


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='llmesr_sasrec',
                    choices=[
                    "llmesr_sasrec", "llmesr_bert4rec", "llmesr_gru4rec",
                    ],
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="yelp", 
                    choices=["yelp", "fashion", "beauty",],  # preprocess by myself
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="inter",
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--pretrain_dir",
                    type=str,
                    default="sasrec_seq",
                    help="the path that pretrained model saved in")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")
parser.add_argument("--do_test",
                    default=False,
                    action="store_true",
                    help="whehther run the test on the well-trained model")
parser.add_argument("--do_emb",
                    default=False,
                    action="store_true",
                    help="save the user embedding derived from the SRS model")
parser.add_argument("--do_group",
                    default=False,
                    action="store_true",
                    help="conduct the group test")
parser.add_argument("--keepon",
                    default=False,
                    action="store_true",
                    help="whether keep on training based on a trained model")
parser.add_argument("--keepon_path",
                    type=str,
                    default="normal",
                    help="the path of trained model for keep on training")
parser.add_argument("--clip_path",
                    type=str,
                    default="",
                    help="the path to save the CLIP-pretrained embedding and adapter")
parser.add_argument("--emb_path",
                    type=str,
                    default="",
                    help="(E25 embedding-encoder ablation) path to the frozen LLM item embedding to load "
                         "as the semantic view: a directory containing item.npy (+ optional pca64.npy), or "
                         "a direct .npy/.pkl file. Empty (default) = data/<dataset>/handled/itm_emb_np.pkl.")
parser.add_argument("--d_llm",
                    type=int,
                    default=0,
                    help="expected dim of the --emb_path embedding; if >0 it is asserted on load "
                         "(ada/small=1536, large/gemini=3072, word2vec=300). 0 = infer from the file.")
parser.add_argument("--ts_user",
                    type=int,
                    default=10,
                    help="the threshold to split the short and long seq")
parser.add_argument("--ts_item",
                    type=int,
                    default=20,
                    help="the threshold to split the long-tail and popular items")

# Model parameters
parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="the hidden size of embedding")
parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="the number of transformer layer")
parser.add_argument("--num_heads",
                    default=1,
                    type=int,
                    help="the number of heads in Trm layer")
parser.add_argument("--num_layers",
                    default=1,
                    type=int,
                    help="the number of GRU layers")
parser.add_argument("--cl_scale",
                    type=float,
                    default=0.1,
                    help="the scale for contastive loss")
parser.add_argument("--mask_crop_ratio",
                    type=float,
                    default=0.3,
                    help="the mask/crop ratio for CL4SRec")
parser.add_argument("--tau",
                    default=1,
                    type=float,
                    help="the temperature for contrastive loss")
parser.add_argument("--sse_ratio",
                    default=0.4,
                    type=float,
                    help="the sse ratio for SSE-PT model")
parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="the dropout rate")
parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="the max length of input sequence")
parser.add_argument("--mask_prob",
                    type=float,
                    default=0.4,
                    help="the mask probability for training Bert model")
parser.add_argument("--aug",
                    default=False,
                    action="store_true",
                    help="whether augment the sequence data")
parser.add_argument("--aug_seq",
                    default=False,
                    action="store_true",
                    help="whether use the augmented data")
parser.add_argument("--aug_seq_len",
                    default=0,
                    type=int,
                    help="the augmented length for each sequence")
parser.add_argument("--aug_file",
                    default="inter",
                    type=str,
                    help="the augmentation file name")
parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="the number of negative samples for training")
parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="the number of negative samples for test")
parser.add_argument("--suffix_num",
                    default=5,
                    type=int,
                    help="the suffix number for augmented sequence")
parser.add_argument("--prompt_num",
                    default=2,
                    type=int,
                    help="the number of prompts")
parser.add_argument("--freeze",
                    default=False,
                    action="store_true",
                    help="whether freeze the pretrained architecture when finetuning")
parser.add_argument("--pg",
                    default="length",
                    choices=['length', 'attention'],
                    type=str,
                    help="choose the prompt generator")
parser.add_argument("--use_cross_att",
                    default=False,
                    action="store_true",
                    help="whether add a cross-attention to interact the dual-view")
parser.add_argument("--adapter_act",
                    default="none",
                    choices=["none", "relu", "gelu", "silu", "tanh"],
                    type=str,
                    help="activation function inserted between the two adapter linear layers "
                         "('none' = pure linear, matches the original paper)")
parser.add_argument("--adapter_layers",
                    default=2,
                    type=int,
                    choices=[1, 2, 3, 4, 5, 6],
                    help="number of linear layers in the adapter (1..6)")
parser.add_argument("--llm_lora_rank",
                    default=0,
                    type=int,
                    help="LoRA rank for the frozen LLM item embedding (semi-unfreeze): "
                         "e_eff = e_frozen + A[item] @ B, A,B trainable, B init 0. 0 (default) = off.")
parser.add_argument("--id_init_path",
                    default="",
                    type=str,
                    help="path to an (item_num x hidden) .pkl used to initialize the collaborative "
                         "item embedding, replacing PCA with a learnable dimensionality reduction "
                         "(e.g. an autoencoder bottleneck). Empty (default) = original pca64 init.")
parser.add_argument("--hard_neg_ratio",
                    default=0.0,
                    type=float,
                    help="probability of drawing each training negative from the K most LLM-similar "
                         "items to the positive (precomputed pool) instead of uniform random. "
                         "0.0 (default) = original random negatives.")
parser.add_argument("--hard_neg_k",
                    default=50,
                    type=int,
                    help="pool size K; loads data/<ds>/handled/hard_neg_k<K>_pool.pkl.")
parser.add_argument("--pop_debias",
                    default=False,
                    action="store_true",
                    help="at test time, sweep inference popularity-debiasing alphas "
                         "(score -= alpha*log1p(item_pop)) and write debias_sweep.csv. "
                         "One trained model -> metrics for all alphas. Eval-only, no training change.")
parser.add_argument("--ssm",
                    default=False,
                    action="store_true",
                    help="use in-batch sampled-softmax loss instead of 1-negative BCE: each valid "
                         "position scores its positive against all other in-batch positives "
                         "(cross-entropy, false negatives masked by item id).")
parser.add_argument("--usr_sem",
                    default=False,
                    action="store_true",
                    help="at test time, sweep a user-semantic scoring term: "
                         "score += gamma * <mean LLM-emb of the user's history items, candidate LLM-emb>. "
                         "Writes usr_sem_sweep.csv. One trained model -> all gamma. Eval-only.")
parser.add_argument("--usr_sem_train",
                    default=False,
                    action="store_true",
                    help="D6 train-time variant: add the user-semantic term to the loss + scoring with a "
                         "LEARNABLE gamma. Uses a causal history centroid during training (no leak).")
parser.add_argument("--usr_sem_train_init",
                    default=0.0,
                    type=float,
                    help="initial value of the learnable gamma for --usr_sem_train (default 0 = baseline at "
                         "init; try ~8 to start near the eval-optimal regime).")
parser.add_argument("--usr_sem_seg",
                    default=False,
                    action="store_true",
                    help="with --usr_sem, also sweep a PER-SEGMENT gamma at eval: apply gamma only to "
                         "short-history (tail) users (seq_len < ts_user). Writes usr_sem_seg_sweep.csv.")
parser.add_argument("--usr_sem_d7",
                    default=False,
                    action="store_true",
                    help="with --usr_sem, also evaluate richer user profiles (D7): recency-weighted "
                         "centroid, max-pool (multi-interest), and an MoE ensemble of the three. "
                         "Writes usr_sem_sweep_{recency,max,ens}.csv.")
parser.add_argument("--full_rank",
                    default=False,
                    action="store_true",
                    help="(D) full-catalog ranking eval: rank the target against ALL items instead of "
                         "100 sampled negatives, with the max-pool D6 term swept. Writes fullrank_sweep.csv.")
parser.add_argument("--moe_diag",
                    default=False,
                    action="store_true",
                    help="Stage-1 two-specialist diagnostic (SASRec): score each test example under "
                         "collab / semantic / full / full+max experts, break down over the 2x2 user x item "
                         "grid, and check whether per-cell routing beats the single best model. Writes moe_diag.csv.")
parser.add_argument("--adapter_hidden_dim",
                    default=-1,
                    type=int,
                    help="intermediate (hidden) dimension of the MLP adapter. "
                         "-1 (default) = in_dim // 2. "
                         "Ignored when --adapter_layers 1.")
parser.add_argument("--adapter_residual",
                    default=False,
                    action="store_true",
                    help="wrap the MLP adapter with a learned linear skip connection: "
                         "output = MLP(x) + Linear_skip(x). "
                         "Ignored when --adapter_layers 1.")
parser.add_argument("--adapter_init_scale",
                    default=1.0,
                    type=float,
                    help="multiply the adapter output-layer init by this factor "
                         "(init-fragility probe; 1.0 = original)")
parser.add_argument("--co_adapter_dim",
                    default=-1,
                    type=int,
                    help="collaborative-side adapter: id embedding dim d_co (init from pca{d_co}), "
                         "projected to hidden_size by a linear adapter. -1 (default) = off / original 64-d.")
parser.add_argument("--pop_gate",
                    default="none",
                    choices=["none", "pop", "item"],
                    type=str,
                    help="popularity-gated dual-view fusion: score = 2[g*co + (1-g)*se] per candidate "
                         "item. 'pop'=g=sigmoid(w*logpop+b) (2 params); 'item'=per-item learnable gate; "
                         "'none' (default)=original equal fusion. Init g=0.5 => identical to baseline.")
parser.add_argument("--alpha",
                    default=0.1,
                    type=float,
                    help="the weight of auxiliary loss")
parser.add_argument("--user_sim_func",
                    default="kd",
                    type=str,
                    help="the type of user similarity function to derive the loss")
parser.add_argument("--item_reg",
                    default=False,
                    action="store_true",
                    help="whether regularize the item embedding by CL")
parser.add_argument("--beta",
                    default=0.1,
                    type=float,
                    help="the weight of regulation loss")
parser.add_argument("--sim_user_num",
                    default=10,
                    type=int,
                    help="the number of similar users for enhancement")
parser.add_argument("--sim_pool",
                    default="mean",
                    type=str,
                    help="self-distillation similar-user pooling: 'mean' (original) or 'weighted' (E10)")
parser.add_argument("--sampling_method",
                    default="topn",
                    type=str,
                    help="self-distillation teacher sampling: 'topn' (original) or 'stochastic' (E11/E12)")
parser.add_argument("--sim_user_file",
                    default="sim_user_100.pkl",
                    type=str,
                    help="teacher-neighbour file (e.g. sim_user_hybrid_100.pkl for hybrid retrieval, E13)")
parser.add_argument("--split_backbone",
                    default=False,
                    action="store_true",
                    help="whether use a split backbone")
parser.add_argument("--co_view",
                    default=False,
                    action="store_true",
                    help="only use the collaborative view")
parser.add_argument("--se_view",
                    default=False,
                    action="store_true",
                    help="only use the semantic view")
parser.add_argument("--id_random_init",
                    default=False,
                    action="store_true",
                    help="randomly initialize the collaborative item embedding instead of PCA LLM embedding (Table-2 ablation)")


# Other parameters
parser.add_argument("--train_batch_size",
                    default=512,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=100,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=20,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="which metric is used to select model.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")

# torch.autograd.set_detect_anomaly(True)  # DISABLED: debug-only NaN tracing, ~2.18x slower backward, no effect on results

args = parser.parse_args()
set_seed(args.seed) # fix the random seed
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.pretrain_dir = os.path.join(args.output_dir, args.pretrain_dir)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.keepon_path = os.path.join(args.output_dir, args.keepon_path)
args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    if args.model_name in ['llmesr_gru4rec']:
        generator = GeneratorAllUser(args, logger, device)
    elif args.model_name in ["llmesr_bert4rec"]:
        generator = BertGeneratorAllUser(args, logger, device)
    elif args.model_name in ["llmesr_sasrec"]:
        generator = Seq2SeqGeneratorAllUser(args, logger, device)
    else:
        raise ValueError

    trainer = SeqTrainer(args, logger, writer, device, generator)

    if args.do_test:
        trainer.test()
    elif args.do_emb:
        trainer.save_user_emb()
    elif args.do_group:
        trainer.test_group()
    else:
        trainer.train()

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()



