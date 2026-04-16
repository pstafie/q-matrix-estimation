# qem_dina

EM algorithm for the DINA model with **constrained Q-matrix** estimation via a
Gurobi MILP. Encodes completeness, distinctness, repetition, optional
hierarchies, and lexicographic symmetry breaking. Includes a generalized-EM
accept/reject gate for monotonicity when the MILP is time-limited.

## Install (editable)

```bash
pip install -e .
```

## Quick start

```python
from qem_dina import DINAEM, EMConfig, QMipConfig
out = DINAEM().fit(R, K, em_cfg=EMConfig(max_iter=100), q_cfg=QMipConfig(include_identity=True))
```

See `examples/run_synthetic.py` for an end‑to‑end demo.