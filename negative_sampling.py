import numpy as np

def negsamp_vectorized_bsearch_preverif(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    reference: https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds


def generate_negative_samples(train_matrix, num_neg=3, num_sets=10):
    neg_samples = []
    for user_id, row in enumerate(train_matrix):
        pos_ind = row.indices
        neg_sample = negsamp_vectorized_bsearch_preverif(pos_ind, train_matrix.shape[1], num_neg * num_sets)
        neg_samples.append(neg_sample)

    return np.asarray(neg_samples).reshape(num_sets, train_matrix.shape[0], num_neg)