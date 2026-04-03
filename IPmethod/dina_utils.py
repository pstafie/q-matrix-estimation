"""
DINA Model Core Utilities
- Data generation
- Recovery metrics (eMRR, mMRR)
- Shared helper functions (asbinary, simA, reorder, delta)
"""

import numpy as np
from scipy.stats import norm
from itertools import permutations


def asbinary(m, K):
    """Convert integer m to binary vector of length K."""
    bits = np.zeros(K, dtype=np.float64)
    for k in range(K - 1, -1, -1):
        bits[k] = m % 2
        m = m // 2
    return bits


def simA(K):
    """Generate all 2^K attribute patterns (L x K matrix)."""
    L = 2 ** K
    A = np.zeros((L, K), dtype=np.float64)
    for m in range(1, L):
        A[m, :] = asbinary(m, K)
    return A


def delta(D):
    """Element-wise recovery rate: 1 - mean(|D|)."""
    return 1.0 - np.mean(np.abs(D))


def reorder_Q(C, A, K):
    """
    Find the best column permutation of A to match target C.
    Tries all K! permutations.
    """
    best_val = -np.inf
    best_A = A.copy()
    for p in permutations(range(K)):
        Ap = A[:, list(p)]
        v = np.sum(C * Ap)
        if v > best_val:
            best_val = v
            best_A = Ap.copy()
    return best_A


def simulate_DINA_data(Q, N, slip=0.2, guess=0.2, rho=0.0):
    """
    Simulate DINA model response data.

    Parameters
    ----------
    Q    : (J, K) binary matrix — true Q-matrix
    N    : int — number of students
    slip : float — P(wrong | capable)
    guess: float — P(right | incapable)
    rho  : float — correlation among attributes

    Returns
    -------
    dict with 'Y' (N x J response matrix), 'alpha' (N x K attribute profiles)
    """
    J, K = Q.shape

    if rho == 0:
        alpha = np.random.binomial(1, 0.5, size=(N, K)).astype(np.float64)
    else:
        Sigma = np.full((K, K), rho)
        np.fill_diagonal(Sigma, 1.0)
        Z = np.random.multivariate_normal(np.zeros(K), Sigma, size=N)
        alpha = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            threshold = norm.ppf((k + 1) / (K + 1))
            alpha[:, k] = (Z[:, k] >= threshold).astype(np.float64)

    natt = Q.sum(axis=1)
    eta = (alpha @ Q.T == natt[np.newaxis, :]).astype(np.float64)
    prob = np.where(eta == 1, 1.0 - slip, guess)
    Y = np.random.binomial(1, prob).astype(np.float64)

    return {"Y": Y, "alpha": alpha, "eta": eta}


def compute_recovery(Qtrue, Qest, K):
    """
    Compute element-wise recovery after best column permutation.

    Returns float — element-wise recovery rate (1.0 = perfect)
    """
    C = np.where(Qtrue == 1.0, 0.8, 0.2)
    Qest_reordered = reorder_Q(C, Qest, K)
    Qest_round = np.round(Qest_reordered)
    return delta(Qtrue - Qest_round)


def get_Q_true_K3J10():
    """K=3, J=10 true Q-matrix from Oka & Okada (2023)."""
    Q = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.float64)
    return Q
