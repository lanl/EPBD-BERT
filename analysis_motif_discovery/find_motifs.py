import os
import sys

home_dir = ""
module_path = os.path.abspath(os.path.join(home_dir))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd

import utility.pickle_utils as pickle_utils
from analysis_motif_finding.motif_utils import (
    find_high_attention,
    merge_motifs,
    make_window,
)  # , motifs_hypergeom_test, filter_motifs,


# args
min_len = 4
min_times_cons = 50
return_idx = False
align_all_ties = True
window_size = 24
min_n_motif = 3

from utility.dnabert2 import get_dnabert2_tokenizer

# loading things
tokenizer = get_dnabert2_tokenizer(max_num_tokens=512, home_dir=home_dir)
# print(tokenizer)

avg_crossattn_weights = pickle_utils.load(
    home_dir + "analysis/weights/all_test_seq_avg_crossattn_weights_list.pkl"
)
print(len(avg_crossattn_weights))

data_df = pd.read_csv(
    home_dir + "data/train_val_test/peaks_with_labels_test.tsv.gz",
    compression="gzip",
    sep="\t",
)
print(data_df.shape)

seq_dict = pickle_utils.load(
    home_dir + "data/processed/seq_with_flanks_dict.pkl"
)  # all seqs
print(len(seq_dict))

neg_seq_dict = pickle_utils.load(
    home_dir + "data/processed/seq_neg_with_flanks_dict.pkl"
)
neg_seqs = [v for k, v in neg_seq_dict.items()]
len(neg_seqs), len(neg_seqs[0])

motif_seqs = {}
pos_seqs = []
for i in range(data_df.shape[0]):
    # getting seq
    x = data_df.loc[i]
    chrom, start, end = x["chrom"], int(x["start"]), int(x["end"])
    seq_id = f"{chrom}_{str(start)}_{str(end)}"
    seq = seq_dict[seq_id]
    # print(len(seq))

    # pos_seq = seq[400:400+200] # need to keep the whole since they are tokenized
    pos_seqs.append(seq)

    toked = tokenizer(
        seq,
        return_tensors="pt",
        padding="longest",
        max_length=512,
        truncation=True,
    )
    # print(toked["input_ids"][0])
    toked_seq_list = tokenizer.decode(toked["input_ids"][0]).split()
    toked_seq_list = toked_seq_list[1:-1]  # removing cls and sep token
    # print(toked_seq_list)
    # print(len(toked_seq_list))

    # cross-attn-weights of the seq from model
    cross_attn_w = avg_crossattn_weights[i]
    cross_attn_w = cross_attn_w[1:-1]
    # print(cross_attn_w.shape)

    expanded_cross_attn_w = []
    for j, w in enumerate(cross_attn_w):
        seq_token_len = len(toked_seq_list[j])
        expanded_cross_attn_w += [w] * seq_token_len
    expanded_cross_attn_w = np.array(expanded_cross_attn_w)
    # print(expanded_cross_attn_w.shape)
    # print(expanded_cross_attn_w)

    # epbd_features = get_epbd_features(f"{seq_id}.pkl")
    # print(cross_attn_w.shape, epbd_features.shape)
    # flip = np.array(epbd_features[5])

    # finding motif regions
    motif_regions = find_high_attention(
        expanded_cross_attn_w,
        min_len=min_len,
        min_times_cons=min_times_cons,
    )
    # motif_regions = find_high_attention(flip, min_len=min_len)
    # print(motif_regions)

    # print(motif_idx[0], motif_idx[1])
    # print(cross_attn_w[motif_idx[0]:motif_idx[1]+1]) # do not need to include this 1
    # plt.plot(cross_attn_w)

    for motif_idx in motif_regions:
        # seq = toked_seq_list[motif_idx[0]:motif_idx[1]]
        # seq = "".join(seq)
        # # print(seq, len(seq))

        # start_pos_in_real_seq = len("".join(toked_seq_list[1:motif_idx[0]]))
        # end_pos_in_real_seq = len("".join(toked_seq_list[1:motif_idx[1]]))
        # print(start_pos_in_real_seq, end_pos_in_real_seq, end_pos_in_real_seq-start_pos_in_real_seq, len(seq))

        start_pos_in_real_seq = motif_idx[0]
        end_pos_in_real_seq = motif_idx[1]
        seq = seq[start_pos_in_real_seq:end_pos_in_real_seq]

        if motif_idx[0] >= motif_idx[1]:
            raise Exception(
                f"Starting motif index ({motif_idx[0]}) can never be ending motif index ({motif_idx[1]})"
            )

        if seq not in motif_seqs:
            motif_seqs[seq] = {
                "seq_idx": [i],
                "atten_region_pos": [(start_pos_in_real_seq, end_pos_in_real_seq)],
            }
        else:
            motif_seqs[seq]["seq_idx"].append(i)
            motif_seqs[seq]["atten_region_pos"].append(
                (start_pos_in_real_seq, end_pos_in_real_seq)
            )

    if i % 10000 == 0:
        print(i, len(motif_seqs))
    # if i==10: break
    # break


print(len(pos_seqs), len(neg_seqs))
print(len(pos_seqs[0]), len(neg_seqs[0]))
print(len(motif_seqs))
# motif_seqs


def count_motif_instances(seqs, motifs, allow_multi_match=False):
    """
    Use Aho-Corasick algorithm for efficient multi-pattern matching
    between input sequences and motif patterns to obtain counts of instances.

    Arguments:
    seqs -- list, numpy array or pandas series of DNA sequences
    motifs -- list, numpy array or pandas series, a collection of motif patterns
        to be matched to seqs

    Keyword arguments:
    allow_multi_match -- bool, whether to allow for counting multiple matchs (default False)

    Returns:
    motif_count -- count of motif instances (int)

    """
    import ahocorasick
    from operator import itemgetter

    motif_count = {}

    A = ahocorasick.Automaton()
    for idx, key in enumerate(motifs):
        A.add_word(key, (idx, key))
        motif_count[key] = 0
    A.make_automaton()

    for i, seq in enumerate(seqs):
        matches = sorted(map(itemgetter(1), A.iter(seq)))
        matched_seqs = []
        for match in matches:
            match_seq = match[1]
            assert match_seq in motifs
            if allow_multi_match:
                motif_count[match_seq] += 1
            else:  # for a particular seq, count only once if multiple matches were found
                if match_seq not in matched_seqs:
                    motif_count[match_seq] += 1
                    matched_seqs.append(match_seq)
        if i % 1000 == 0:
            print(i)

    return motif_count


def motifs_hypergeom_test(
    pos_seqs,
    neg_seqs,
    motifs,
    p_adjust="fdr_bh",
    alpha=0.05,
    verbose=False,
    allow_multi_match=False,
    **kwargs,
):
    """
    Perform hypergeometric test to find significantly enriched motifs in positive sequences.
    Returns a list of adjusted p-values.

    Arguments:
    pos_seqs -- list, numpy array or pandas series of positive DNA sequences
    neg_seqs -- list, numpy array or pandas series of negative DNA sequences
    motifs -- list, numpy array or pandas series, a collection of motif patterns
        to be matched to seqs

    Keyword arguments:
    p_adjust -- method used to correct for multiple testing problem. Options are same as
        statsmodels.stats.multitest (default 'fdr_bh')
    alpha -- cutoff FDR/p-value to declare statistical significance (default 0.05)
    verbose -- verbosity argument (default False)
    allow_multi_match -- bool, whether to allow for counting multiple matchs (default False)

    Returns:
    pvals -- a list of p-values.

    """
    from scipy.stats import hypergeom
    import statsmodels.stats.multitest as multi

    pvals = []
    N = len(pos_seqs) + len(neg_seqs)
    K = len(pos_seqs)
    print("counting motif seqs in all seqs")
    motif_count_all = count_motif_instances(
        pos_seqs + neg_seqs, motifs, allow_multi_match=allow_multi_match
    )
    print("counting motif seqs in pos seqs")
    motif_count_pos = count_motif_instances(
        pos_seqs, motifs, allow_multi_match=allow_multi_match
    )

    for i, motif in enumerate(motifs):
        print(i)
        n = motif_count_all[motif]
        x = motif_count_pos[motif]
        pval = hypergeom.sf(x - 1, N, K, n)
        if verbose:
            if pval < 1e-5:
                print(
                    "motif {}: N={}; K={}; n={}; x={}; p={}".format(
                        motif, N, K, n, x, pval
                    )
                )
        #         pvals[motif] = pval
        # if i % 1000 == 0:

        pvals.append(pval)

    # adjust p-value
    if p_adjust is not None:
        pvals = list(multi.multipletests(pvals, alpha=alpha, method=p_adjust)[1])
    return pvals


pvals_saved_file = (
    home_dir
    + f"analysis_motif_finding/p_values/pvalues_list_min_times_cons_{min_times_cons}.pkl"
)
if os.path.exists(pvals_saved_file):
    pvals = pickle_utils.load(pvals_saved_file)
else:
    pvals = motifs_hypergeom_test(pos_seqs, neg_seqs, list(motif_seqs.keys()))
    pickle_utils.save(pvals, pvals_saved_file)
len(pvals)


# pvals statistics
print(np.min(pvals), np.max(pvals), np.median(pvals))

# motifs_to_keep = filter_motifs(pos_seqs, neg_seqs, list(motif_seqs.keys()), cutoff=np.median(pvals), return_idx=return_idx)
motifs = list(motif_seqs.keys())
pval_cutoff = 0.005  # np.median(pvals)
motifs_to_keep = [motifs[i] for i, pval in enumerate(pvals) if pval <= pval_cutoff]
kept_motif_seqs = {k: motif_seqs[k] for k in motifs_to_keep}
print(len(kept_motif_seqs))
kept_motif_seqs_copy = kept_motif_seqs.copy()

merged_motif_seqs = merge_motifs(
    kept_motif_seqs_copy, min_len=min_len, align_all_ties=True
)
print(len(merged_motif_seqs))
# merged_motif_seqs

new_merged_motif_seqs = make_window(
    merged_motif_seqs, pos_seqs, window_size=window_size
)  # not correct
print(len(new_merged_motif_seqs))
new_merged_motif_seqs

new_new_merged_motif_seqs = {
    k: coords
    for k, coords in new_merged_motif_seqs.items()
    if len(coords["seq_idx"]) >= min_n_motif
}
print(len(new_new_merged_motif_seqs))

from Bio import motifs
from Bio.Seq import Seq

save_file_dir = (
    home_dir
    + f"analysis_motif_finding/motifs_job_min_times_cons={min_times_cons}_pval_cutoff={pval_cutoff:.3f}/"
)
if save_file_dir is not None:
    os.makedirs(save_file_dir, exist_ok=True)
    for motif_no, (motif, instances) in enumerate(new_new_merged_motif_seqs.items()):
        # saving to files
        with open(
            save_file_dir
            + "/motif_{}_{}_{}.txt".format(
                motif_no + 1, motif, len(instances["seq_idx"])
            ),
            "w",
        ) as f:
            for seq in instances["seqs"]:
                f.write(seq + "\n")
        # make weblogo
        seqs = [Seq(v) for i, v in enumerate(instances["seqs"])]
        # print(seqs)
        m = motifs.create(seqs)
        # m.weblogo(save_file_dir+"/motif_{}_{}_weblogo.png".format(motif, len(instances['seq_idx'])), format='png_print',
        #                     show_fineprint=False, show_ends=False, color_scheme='color_classic')
