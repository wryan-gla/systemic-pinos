"""POD-DeepONet for a single vessel."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense  # type: ignore

from .base import VesselOperator


def compute_pod_basis(snapshots, num_modes):
    """POD basis of standardised training fields.

    snapshots: [n_samples, n_t, n_x] array of one output (flow or pressure),
               already standardised.
    Returns (basis [num_modes, n_t * n_x], coeff_mean, coeff_std); the
    coefficient statistics are used to standardise the branch net's targets.
    """
    flat = snapshots.reshape(len(snapshots), -1)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    basis = vt[:num_modes]
    coeffs = flat @ basis.T
    coeff_std = coeffs.std(0)
    return basis, coeffs.mean(0), np.where(coeff_std < 1e-8, 1., coeff_std)


class PODDeepONet(VesselOperator):
    """DeepONet whose trunk is a fixed POD basis computed from training data.

    A branch MLP per output maps (inflow, parameters) to POD coefficients,
    which are combined with the basis to give the full n_t x n_x field. The
    PDE residual is taken by finite differences on that grid.

    flow_pod, pressure_pod: outputs of `compute_pod_basis`. They are stored as
    non-trainable weights, so they are saved and restored with the model.
    """

    def __init__(self, par_idx, flow_pod, pressure_pod, branch_units=(192, 160),
                 activation="gelu", **kwargs):
        super().__init__(par_idx, **kwargs)
        self.branches, self.pods = [], []
        for name, pod in (("flow", flow_pod), ("pressure", pressure_pod)):
            num_modes = pod[0].shape[0]
            self.branches.append(tf.keras.Sequential(
                [Dense(u, activation=activation) for u in branch_units] + [Dense(num_modes)]))
            stored = []
            for array, label in zip(pod, ("basis", "coeff_mean", "coeff_std")):
                w = self.add_weight(shape=array.shape, initializer="zeros", trainable=False,
                                    name=f"{name}_{label}")
                w.assign(array.astype(np.float32))
                stored.append(w)
            self.pods.append(stored)

    def call(self, inputs, physical=False):
        inflow, pars = inputs
        z = tf.concat([self.inflow_norm(inflow), self.par_norm(pars)], -1)
        fields = []
        for branch, (basis, c_mean, c_std) in zip(self.branches, self.pods):
            coeffs = branch(z) * c_std + c_mean
            fields.append(tf.reshape(coeffs @ basis, [-1, self.n_t, self.n_x]))
        out = tf.stack(fields, -1)
        return self.to_physical(out) if physical else out
