import numpy as np

from scipy.spatial.distance import cdist
import ot

def kl_divergence(p, q, bins=40):
    """
    Compute the KL divergence between two 1D distributions p and q.
    p, q: numpy arrays (samples)
    bins: number of bins for histogram estimation
    Returns: KL divergence (float)
    """
    p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
    p_hist = np.clip(p_hist, 1e-10, None)
    q_hist = np.clip(q_hist, 1e-10, None)
    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)
    return np.sum(p_hist * np.log(p_hist / q_hist))


def calculate_w1_distance_for_eta(actual_df, gen_df, p_cols, q_cols, v_cols, theta_cols):
    """
    Calculate the Wasserstein-1 (W1) distance between actual and generated dataframes for P, Q, V, Theta columns.
    Theta columns are converted to radians before calculation.
    Returns the W1 distance (float).
    """
    cols = p_cols + q_cols + v_cols + theta_cols
    actual_df = actual_df.copy()
    gen_df = gen_df.copy()
    actual_df[theta_cols] = np.deg2rad(actual_df[theta_cols])
    gen_df[theta_cols] = np.deg2rad(gen_df[theta_cols])
    data_gt = actual_df[cols].values
    data_syn = gen_df[cols].values
    n = len(data_gt)
    m = len(data_syn)
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = cdist(data_gt, data_syn, metric='euclidean')
    wasserstein_dist = ot.emd2(a, b, M)
    return wasserstein_dist
