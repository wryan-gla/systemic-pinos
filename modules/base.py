"""Plumbing shared by the per-vessel operators."""
import tensorflow as tf

from . import physics


class VesselOperator(tf.keras.Model):
    """Base class for an operator emulating flow and pressure in one vessel.

    Maps an inflow waveform [batch, n_t] and a parameter vector [batch, n_pars]
    to flow and pressure on an n_t x n_x space-time grid, [batch, n_t, n_x, 2].
    Time covers one period, t in [0, 1); space runs inlet to outlet, x in [0, 1].

    Outputs are standardised with the given flow/pressure means and stds; pass
    `physical=True` when calling to get them in physical units. Call
    `adapt(inflows, pars)` on training data before training.

    par_idx:  where this vessel's length, radius, period and stiffness sit in
              the parameter vector (see `physics.py`).
    tapering: use an exponentially tapering radius in the tube law.
    """

    def __init__(self, par_idx, n_t=128, n_x=128,
                 flow_mean=0., flow_std=1., pressure_mean=0., pressure_std=1.,
                 tapering=False, **kwargs):
        super().__init__(**kwargs)
        self.par_idx = par_idx
        self.n_t, self.n_x = n_t, n_x
        self.tapering = tapering
        self.out_mean = tf.constant([flow_mean, pressure_mean], tf.float32)
        self.out_std = tf.constant([flow_std, pressure_std], tf.float32)
        self.inflow_norm = tf.keras.layers.Normalization(axis=None)
        self.par_norm = tf.keras.layers.Normalization(axis=-1)

    def adapt(self, inflows, pars):
        self.inflow_norm.adapt(inflows)
        self.par_norm.adapt(pars)

    def standardise(self, y):
        return (y - self.out_mean) / self.out_std

    def to_physical(self, y):
        return y * self.out_std + self.out_mean

    def junction(self, inflow, pars, end):
        """Physical (q, p) at the "inlet" or "outlet", each [batch, n_t]."""
        out = self((inflow, pars), physical=True)[:, :, 0 if end == "inlet" else -1]
        return out[..., 0], out[..., 1]

    def inflow_loss(self, inflow, pars):
        """Mismatch between the inlet flow and the prescribed inflow."""
        q_in, _ = self.junction(inflow, pars, "inlet")
        return tf.reduce_mean(((q_in - inflow) / self.out_std[0]) ** 2)

    def pi_loss(self, inflow, pars, smooth=False):
        """PDE residual loss, by finite differences on the output grid."""
        out = self((inflow, pars), physical=True)
        vp = physics.vessel_pars(pars, self.par_idx)
        return physics.grid_pi_loss(out[..., 0], out[..., 1], vp, self.tapering, smooth)
