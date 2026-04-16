import numpy as np

def compute_tau(R, s, g, Xi, nu, eps=1e-12):
    """
    R:  (N, J) binary responses
    s,g: (J,) slip/guess
    Xi: (P, J) capability matrix xi_j(alpha) in {0,1}
    nu: (P,) prior over profiles
    Returns: tau of shape (N, P)
    """
    N, J = R.shape
    P = Xi.shape[0]
    s  = np.clip(s, eps, 1 - eps)
    g  = np.clip(g, eps, 1 - eps)
    nu = np.clip(nu, eps, None)

    log1ms = np.log1p(-s)           # log(1 - s)
    logs   = np.log(s)
    logg   = np.log(g)
    log1mg = np.log1p(-g)           # log(1 - g)

    # Per (n,j) itemwise logs for capable vs not capable
    # shape (N,J) each
    Lcap = R * log1ms + (1 - R) * logs
    Lnoc = R * logg   + (1 - R) * log1mg

    B = Lnoc.sum(axis=1)            # (N,)
    Delta = Lcap - Lnoc             # (N,J)

    lognu = np.log(nu)              # (P,)

    # For each n: ell_n = lognu + B[n] + Xi @ Delta[n]
    # We can batch this by looping over n (usually fine) or stack cleverly.
    tau = np.empty((N, P), dtype=float)
    for n in range(N):
        ell = lognu + B[n] + Xi @ Delta[n]   # (P,)
        # log-sum-exp normalization
        m = np.max(ell)
        z = np.exp(ell - m)
        tau[n] = z / z.sum()

    return tau
