"""Physics-informed DeepONet for a single vessel."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense  # type: ignore

from . import physics
from .base import VesselOperator


class ResidualMLP(tf.keras.layers.Layer):
    def __init__(self, width, depth, out_dim, activation="gelu", out_activation=None):
        super().__init__()
        self.inp = Dense(width, activation=activation)
        self.hidden = [Dense(width, activation=activation) for _ in range(depth)]
        self.out = Dense(out_dim, activation=out_activation)

    def call(self, x):
        x = self.inp(x)
        for layer in self.hidden:
            x = x + layer(x)
        return self.out(x)


class DeepONet(VesselOperator):
    """DeepONet with separate flow/pressure branches and a shared trunk.

    Branch nets encode the inflow waveform; the trunk encodes a query point
    (cos 2πt, sin 2πt, x) together with the parameters. Output at each point is
    the dot product of branch and trunk features plus a bias.

    Call with (inflow, pars) for the full n_t x n_x grid, [batch, n_t, n_x, 2],
    or with (inflow, pars, t, x) for arbitrary points, t and x shaped
    [batch or 1, n_points, 1], giving [batch, n_points, 2]. Being mesh-free,
    the PDE residual is taken with automatic differentiation at random points.
    """

    def __init__(self, par_idx, latent_dim=128, branch_units=(128, 128),
                 trunk_width=128, trunk_depth=2, activation="gelu", **kwargs):
        super().__init__(par_idx, **kwargs)
        self.branch_q = tf.keras.Sequential(
            [Dense(u, activation=activation) for u in branch_units] + [Dense(latent_dim)])
        self.branch_p = tf.keras.Sequential(
            [Dense(u, activation=activation) for u in branch_units] + [Dense(latent_dim)])
        self.trunk = ResidualMLP(trunk_width, trunk_depth, latent_dim, activation, out_activation=activation)
        self.bias = self.add_weight(shape=(2,), initializer="zeros", name="bias")

        t = np.arange(self.n_t) / self.n_t
        tt, xx = np.meshgrid(t, np.linspace(0., 1., self.n_x), indexing="ij")
        self.t_line = tf.constant(t.reshape(1, -1, 1), tf.float32)
        self.t_grid = tf.constant(tt.reshape(1, -1, 1), tf.float32)
        self.x_grid = tf.constant(xx.reshape(1, -1, 1), tf.float32)

    def call(self, inputs, physical=False):
        on_grid = len(inputs) == 2
        if on_grid:
            inflow, pars = inputs
            t, x = self.t_grid, self.x_grid
        else:
            inflow, pars, t, x = inputs
        b, n = tf.shape(inflow)[0], tf.shape(t)[1]
        t = tf.broadcast_to(t, [b, n, 1])
        x = tf.broadcast_to(x, [b, n, 1])
        p = tf.broadcast_to(self.par_norm(pars)[:, None, :], [b, n, pars.shape[-1]])

        trunk = self.trunk(tf.concat([tf.cos(2 * np.pi * t), tf.sin(2 * np.pi * t), x, p], -1))
        a = self.inflow_norm(inflow)
        q = tf.einsum("bl,bnl->bn", self.branch_q(a), trunk) + self.bias[0]
        pr = tf.einsum("bl,bnl->bn", self.branch_p(a), trunk) + self.bias[1]
        out = tf.stack([q, pr], -1)

        if on_grid:
            out = tf.reshape(out, [-1, self.n_t, self.n_x, 2])
        return self.to_physical(out) if physical else out

    def junction(self, inflow, pars, end):
        x = tf.fill([1, self.n_t, 1], 0. if end == "inlet" else 1.)
        out = self((inflow, pars, self.t_line, x), physical=True)
        return out[..., 0], out[..., 1]

    def pi_loss(self, inflow, pars, n_points=1024):
        b = tf.shape(inflow)[0]
        t = tf.random.uniform([b, n_points, 1])
        x = tf.random.uniform([b, n_points, 1])
        vp = physics.vessel_pars(pars, self.par_idx)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([t, x])
            out = self((inflow, pars, t, x), physical=True)
            q, p = out[..., :1], out[..., 1:]
            r0 = physics.tapered_radius(x, vp["radius"], vp["length"]) if self.tapering else vp["radius"]
            A = physics.area(p, r0, vp["stiffness"])
            q2A = q ** 2 / A
        # chain rule from unit (t, x) to physical time and distance
        T, L = vp["period"] * physics.Q0, vp["length"]
        mass, momentum = physics.residuals(
            q, A, vp["period"],
            dA_dt=tape.gradient(A, t) / T, dq_dt=tape.gradient(q, t) / T,
            dq_dx=tape.gradient(q, x) / L, dq2A_dx=tape.gradient(q2A, x) / L,
            dp_dx=tape.gradient(p, x) / L)
        del tape
        return tf.reduce_mean(mass ** 2) + tf.reduce_mean(momentum ** 2)
