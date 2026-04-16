# Q-Matrix Estimation

Research project comparing methods for recovering a binary Q-matrix in the DINA model, evaluated on both element-wise and matrix-wise recovery metrics.

## Project Structure

```text
q-matrix-estimation/
├── README.md
├── experiments/
│   └── run_simulation.py
├── methods/
│   ├── inference_methods/
│   │   ├── dina_utils.py
│   │   ├── gibbs_sampling/
│   │   │   └── dina_gibbs_estimator.py
│   │   ├── variational_bayes/
│   │   │   └── dina_vb_estimator.py
│   │   └── lasso_penalized/
│   │       └── dina_lasso_estimator.py
│   │
│   └── ip/
│       ├── core/
│       │   └── dina-qip/
│       ├── scripts/
│       │   ├── run_mip_simulation.py
│       │   └── smoke_test.py
│       ├── jobs/
│       │   └── run_mip_job.sh
│       ├── env/
│       │   └── setup_env.sh
│       └── requirements.txt
└── results/
    ├── logs/
    │   ├── mip_sim_46539311.err
    │   └── mip_sim_46539311.out
    ├── results_mip_simulation.csv
    └── results_mip_simulation.txt
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
