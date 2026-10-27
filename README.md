# Physics-Informed Neural Operators for Multi-vessel Cardiovascular Networks

Code accompanying the paper:

**[Physics-Informed Neural Operators for Parameter Inference in Multi-vessel Cardiovascular Networks](https://link.springer.com/article/10.1007/s10439-026-04219-1)**

---

## Overview

This repository contains basic implementations of the three physics-informed neural operators compared in the paper, for emulating 1D blood flow through a network of arteries:

| Model | Architecture | PDE residual |
|---|---|---|
| `FNO` | Factorised Fourier neural operator on the full space-time grid | Finite differences on the output grid |
| `DeepONet` | Separate flow/pressure branch nets on the inflow; shared residual-MLP trunk on (t, x, parameters) | Automatic differentiation at random collocation points |
| `PODDeepONet` | Branch nets predicting coefficients of a fixed POD basis built from training data | Finite differences on the output grid |

It is intended as a starting point for trying the modelling approach, **not** as a pipeline for reproducing the paper's results. No data or trained weights are included and network sizes are arbitrary.

### Method summary

Each vessel in the network gets its own operator, mapping

```
(inflow waveform q_in(t), parameter vector θ)  →  (Q(t, x), P(t, x))
```

over one cardiac cycle and along the vessel. Each operator is trained on a combination of:

1. **Data loss**: mismatch to simulated flow and pressure fields.
2. **Physics loss**: residuals of the 1D mass and momentum equations, with area recovered from pressure via a nonlinear tube law (optionally with a tapering radius).
3. **Bifurcation loss**: flow conservation and pressure continuity between a parent's outlet and its daughters' inlets.
4. **Inflow condition** for the root vessel. The FNO enforces this exactly; the DeepONets use a penalty.

Loss terms are balanced with gradient-norm-based adaptive weights. Vessels are trained generation by generation, root first: each daughter is fitted with its already-trained parent held fixed in the bifurcation loss. The trained emulators then replace the numerical solver inside MCMC for parameter inference (not included here).

---

## Repository structure

```
systemic-pinos/
├── train.py              # Example training script for a small vessel network
├── requirements.txt
└── modules/
    ├── base.py               # Shared normalisation, output scaling, junction and loss helpers
    ├── fno.py                # Factorised FNO
    ├── deeponet.py           # DeepONet
    ├── pod_deeponet.py       # POD-DeepONet and POD basis computation
    ├── physics.py            # Tube law, PDE residuals, bifurcation loss
    └── adaptive_weights.py   # Gradient-norm loss balancing
```

---

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies: TensorFlow ≥ 2.11, NumPy, tqdm.

---

## Quick start

All three models share the same interface. Inputs are an inflow waveform `[batch, n_t]` and a parameter vector `[batch, n_pars]`; outputs are flow and pressure on an `n_t × n_x` grid, `[batch, n_t, n_x, 2]`.

```python
from modules import FNO

# Where this vessel's length, radius, cardiac period and wall stiffness sit in the parameter vector
par_idx = {"length": 0, "radius": 1, "period": 2, "stiffness": 3}

model = FNO(par_idx, n_t=128, n_x=128, width=32, num_modes=16, num_layers=4,
            flow_mean=q_mean, flow_std=q_std, pressure_mean=p_mean, pressure_std=p_std)
model.adapt(inflows, pars)                            # input normalisation from training data

out = model((inflows, pars))                          # standardised outputs, for the data loss
qp = model((inflows, pars), physical=True)            # flow and pressure in physical units
pde = model.pi_loss(inflows, pars)                    # mass + momentum residual
q_out, p_out = model.junction(inflows, pars, "outlet")  # values at the vessel end, for bifurcation losses
```

`DeepONet` can also be evaluated at arbitrary points with `model((inflows, pars, t, x))`. `PODDeepONet` takes the bases returned by `compute_pod_basis` on standardised training fields.

The physical constants in [`modules/physics.py`](modules/physics.py) follow the scaled units of the solver used in the paper; adjust them if your data uses different units.

---

## Training

[`train.py`](train.py) trains one operator per vessel on a small example network (vessel 1 splitting into vessels 2 and 3):

```bash
python train.py --model fno --data simulations.npz      # or --model deeponet / pod
```

Simulations from a 1D haemodynamics solver are expected as an `.npz` file containing:

| Key | Shape | Contents |
|---|---|---|
| `inflow` | `[N, n_t]` | Inlet flow over one period |
| `pars` | `[N, n_pars]` | Parameter vectors |
| `vessel<k>` | `[N, n_t, n_x, 2]` | Flow and pressure in vessel `k` |
| `pi_inflow`, `pi_pars` | `[M, n_t]`, `[M, n_pars]` | Optional extra inputs, without outputs, for the physics losses |

To use a different network or parameter layout, edit `PARENT`, `PAR_IDX` and `TAPERED` at the top of the script.

---

## Citation

If you use this code, please cite:

```bibtex
@article{ryan2026pino,
  author  = {Ryan, William and Taylor-LaPole, Alyssa and Olufsen, Mette S. and Vyshemirsky, Vladislav and Husmeier, Dirk},
  title   = {Physics-Informed Neural Operators for Parameter Inference in Multi-vessel Cardiovascular Networks},
  journal = {Annals of Biomedical Engineering},
  year    = {2026},
  doi     = {10.1007/s10439-026-04219-1},
  url     = {https://link.springer.com/article/10.1007/s10439-026-04219-1}
}
```

---

## License

MIT
