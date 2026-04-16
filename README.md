# Q-Matrix Estimation

Research project comparing methods for recovering a binary Q-matrix in the DINA model, evaluated on both element-wise and matrix-wise recovery metrics.

## Project Structure

```text
q-matrix-estimation/
├── methods/
│   ├── ip/
│   │   ├── dina-qip/
│   │   ├── run_mip_simulation.py
│   │   ├── run_mip_job.sh
│   │   ├── setup_env.sh
│   │   ├── smoke_test.py
│   │   └── requirements.txt
│   └── mcmc_and_penalized/
│       ├── dina_gibbs_estimator.py
│       ├── dina_lasso_estimator.py
│       ├── dina_vb_estimator.py
│       ├── dina_utils.py
│       └── run_simulation.py
└── results/
```

## Methods

| Method | Folder | Description |
|---|---|---|
| Integer Programming (IP) | methods/ip/ | MIP-based Q-matrix recovery |
| Gibbs Sampling | methods/mcmc_and_penalized/ | MCMC posterior estimation |
| LASSO | methods/mcmc_and_penalized/ | Penalized regression |
| Variational Bayes (VB) | methods/mcmc_and_penalized/ | Variational inference |

## Evaluation Metrics

- Element-wise recovery rate
- Matrix-wise recovery rate

## Setup

```bash
cd methods/ip
bash setup_env.sh
