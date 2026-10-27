"""Train physics-informed neural operators on a network of vessels.

    python train.py --model fno --data simulations.npz      # or deeponet / pod

No data is included in this repository. Supply your own simulations (e.g. from
a 1D haemodynamics solver) as an .npz file containing

    inflow      [N, n_t]            inlet flow over one period
    pars        [N, n_pars]         parameter vectors
    vessel<k>   [N, n_t, n_x, 2]    flow and pressure in vessel k, for every
                                    vessel in PARENT below
    pi_inflow   [M, n_t]            optional unlabelled inputs for the physics
    pi_pars     [M, n_pars]         losses (default: the training inputs)

Describe your network by editing PARENT, PAR_IDX and TAPERED below. As in the
paper, one operator is trained per vessel, generation by generation: each
daughter is fitted to its data and the PDE residual, and coupled to its
(already trained) parent and its sister through a bifurcation loss.
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from modules import FNO, DeepONet, PODDeepONet, compute_pod_basis, bifurcation_loss, AdaptiveLossWeights

# Each vessel's parent (None for the root). Example: vessel 1 splits into 2 and 3.
PARENT = {1: None, 2: 1, 3: 1}
# Where each vessel's length, radius, period and stiffness sit in `pars`.
# Example layout: [L1, L2, L3, r1, r2, r3, period, stiffness].
PAR_IDX = {v: {"length": v - 1, "radius": v + 2, "period": 6, "stiffness": 7} for v in PARENT}
# Vessels with an exponentially tapering radius.
TAPERED = set()


def generations():
    depth = lambda v: 0 if PARENT[v] is None else depth(PARENT[v]) + 1
    gens = {}
    for v in PARENT:
        gens.setdefault(depth(v), []).append(v)
    return [gens[d] for d in sorted(gens)]


def sister(v):
    return next(s for s in PARENT if s != v and PARENT[s] == PARENT[v])


def build_model(kind, v, y, n_t, n_x):
    """Operator for vessel v; y is its training output [N, n_t, n_x, 2]. Sizes are arbitrary."""
    stats = dict(flow_mean=y[..., 0].mean(), flow_std=y[..., 0].std(),
                 pressure_mean=y[..., 1].mean(), pressure_std=y[..., 1].std())
    common = dict(par_idx=PAR_IDX[v], n_t=n_t, n_x=n_x, tapering=v in TAPERED,
                  **{k: float(s) for k, s in stats.items()})
    if kind == "fno":
        return FNO(width=32, num_modes=16, num_layers=4, inlet=PARENT[v] is None, **common)
    if kind == "deeponet":
        return DeepONet(latent_dim=128, branch_units=(128, 128), trunk_width=128, trunk_depth=2, **common)
    flow_pod = compute_pod_basis((y[..., 0] - stats["flow_mean"]) / stats["flow_std"], num_modes=64)
    pressure_pod = compute_pod_basis((y[..., 1] - stats["pressure_mean"]) / stats["pressure_std"], num_modes=64)
    return PODDeepONet(flow_pod=flow_pod, pressure_pod=pressure_pod, branch_units=(192, 160), **common)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fno", "deeponet", "pod"], default="fno")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--update-every", type=int, default=50, help="steps between loss-weight updates")
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    data = np.load(args.data)
    inflow = data["inflow"].astype(np.float32)
    pars = data["pars"].astype(np.float32)
    outputs = {v: data[f"vessel{v}"].astype(np.float32) for v in PARENT}
    pi_inflow = data["pi_inflow"].astype(np.float32) if "pi_inflow" in data.files else inflow
    pi_pars = data["pi_pars"].astype(np.float32) if "pi_pars" in data.files else pars
    n, n_t = inflow.shape
    n_x = outputs[1].shape[2]

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    val, train = idx[:int(args.val_frac * n)], idx[int(args.val_frac * n):]
    bs = args.batch_size
    steps_per_epoch = len(train) // bs

    models = {}
    for v in PARENT:
        models[v] = build_model(args.model, v, outputs[v][train], n_t, n_x)
        models[v].adapt(inflow[train], pars[train])
        models[v]((inflow[:1], pars[:1]))  # build weights eagerly

    def vessel_losses(v, batch, pi_batch):
        model = models[v]
        x_in, x_pars, y = batch
        pi_in, pi_par = pi_batch
        losses = {"data": tf.reduce_mean((model((x_in, x_pars)) - model.standardise(y)) ** 2),
                  "physics": model.pi_loss(pi_in, pi_par)}
        if PARENT[v] is None:
            losses["inflow"] = model.inflow_loss(pi_in, pi_par)
        else:
            losses["bifurcation"] = bifurcation_loss(
                models[PARENT[v]].junction(pi_in, pi_par, "outlet"),
                model.junction(pi_in, pi_par, "inlet"),
                models[sister(v)].junction(pi_in, pi_par, "inlet"))
        return losses

    def make_step(v, optimizer):
        variables = models[v].trainable_variables

        @tf.function
        def step(batch, pi_batch, weights):
            with tf.GradientTape() as tape:
                losses = vessel_losses(v, batch, pi_batch)
                total = tf.add_n([weights[k] * losses[k] for k in losses])
            optimizer.apply_gradients(zip(tape.gradient(total, variables), variables))
            return losses
        return step

    def grad_norms(v, batch, pi_batch):
        variables = models[v].trainable_variables
        with tf.GradientTape(persistent=True) as tape:
            losses = vessel_losses(v, batch, pi_batch)
        return {k: float(tf.linalg.global_norm(tape.gradient(loss, variables, unconnected_gradients="zero")))
                for k, loss in losses.items()}

    def val_mse(v):
        errs = [tf.reduce_mean((models[v]((inflow[b], pars[b])) - models[v].standardise(outputs[v][b])) ** 2)
                for b in np.array_split(val, max(1, len(val) // bs))]
        return float(np.mean(errs))

    for gen in generations():
        steps, weights = {}, {}
        for v in gen:
            schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, args.epochs * steps_per_epoch, alpha=0.1)
            optimizer = tf.keras.optimizers.Adam(schedule)
            optimizer.build(models[v].trainable_variables)
            steps[v] = make_step(v, optimizer)
            weights[v] = AdaptiveLossWeights(["data", "physics", "inflow" if PARENT[v] is None else "bifurcation"])

        it = 0
        pbar = tqdm(range(args.epochs), desc=f"vessels {gen}")
        for _ in pbar:
            perm = rng.permutation(train)
            for i in range(steps_per_epoch):
                b = perm[i * bs:(i + 1) * bs]
                pb = rng.integers(len(pi_inflow), size=bs)
                pi_batch = (tf.constant(pi_inflow[pb]), tf.constant(pi_pars[pb]))
                for v in gen:
                    batch = (tf.constant(inflow[b]), tf.constant(pars[b]), tf.constant(outputs[v][b]))
                    if it % args.update_every == 0:
                        weights[v].update(grad_norms(v, batch, pi_batch))
                    losses = steps[v](batch, pi_batch, weights[v].as_tensors())
                it += 1
            postfix = {k: f"{float(l):.2e}" for k, l in losses.items()}
            postfix.update({f"val{v}": f"{val_mse(v):.2e}" for v in gen})
            pbar.set_postfix(postfix)

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            for v in gen:
                models[v].save_weights(os.path.join(args.save_dir, f"{args.model}_vessel{v}.weights.h5"))


if __name__ == "__main__":
    main()
