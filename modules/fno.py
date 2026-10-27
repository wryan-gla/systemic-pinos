"""Factorised Fourier neural operator for a single vessel."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense  # type: ignore

from .base import VesselOperator


class FNO(VesselOperator):
    """Physics-informed FNO mapping (inflow, parameters) to (q, p) on a grid.

    The inflow, time features (cos 2πt, sin 2πt), x and the parameters are
    stacked into a channel vector at every grid point, lifted to `width`
    channels, passed through `num_layers` Fourier layers and projected to (q, p).

    Each Fourier layer is factorised: separate truncated spectral convolutions
    along t and along x are summed, mixed by a pointwise two-layer MLP, and
    added back on as a residual. Time is periodic so it uses a plain FFT; the
    spatial signal is mirrored first so the transform does not see a jump
    between inlet and outlet.

    inlet: hard-enforce q(t, 0) = inflow(t) (for the root vessel).
    """

    def __init__(self, par_idx, width=32, num_modes=16, num_layers=4,
                 lift_layers=2, lift_units=128, inlet=False, **kwargs):
        super().__init__(par_idx, **kwargs)
        self.num_modes = num_modes
        self.inlet = inlet
        self.lift = tf.keras.Sequential(
            [Dense(lift_units, activation="gelu") for _ in range(lift_layers)] + [Dense(width)])
        self.proj = Dense(2)
        # complex spectral weights stored as [real/imag, modes, in, out]
        spectral_weight = lambda name: self.add_weight(
            shape=(2, num_modes, width, width), name=name,
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
        self.w_t = [spectral_weight(f"w_t{i}") for i in range(num_layers)]
        self.w_x = [spectral_weight(f"w_x{i}") for i in range(num_layers)]
        self.mix = [tf.keras.Sequential([Dense(4 * width, activation="gelu"), Dense(width, activation="gelu")])
                    for _ in range(num_layers)]

        t = np.arange(self.n_t) / self.n_t
        x = np.linspace(0., 1., self.n_x)
        tt, xx = np.meshgrid(t, x, indexing="ij")
        self.coords = tf.constant(np.stack(
            [np.cos(2 * np.pi * tt), np.sin(2 * np.pi * tt), xx], -1)[None], tf.float32)  # [1, n_t, n_x, 3]
        self.x = tf.constant(x[None, None, :], tf.float32)

    def spectral_conv(self, v, w, axis):
        """Truncated Fourier multiplier along t (axis=1) or x (axis=2) of v [batch, n_t, n_x, width]."""
        if axis == 1:
            u, n = tf.transpose(v, [0, 2, 3, 1]), self.n_t
        else:
            u, n = tf.transpose(v, [0, 1, 3, 2]), self.n_x
            u = tf.concat([u, u[..., ::-1]], -1)  # mirror
        fft_len = u.shape[-1]
        u_hat = tf.signal.rfft(u)[..., :self.num_modes]
        u_hat = tf.einsum("bkim,mio->bkom", u_hat, tf.complex(w[0], w[1]))
        u = tf.signal.irfft(u_hat, fft_length=[fft_len])[..., :n]
        return tf.transpose(u, [0, 3, 1, 2] if axis == 1 else [0, 1, 3, 2])

    def call(self, inputs, physical=False):
        inflow, pars = inputs
        shape = [tf.shape(inflow)[0], self.n_t, self.n_x]
        a = tf.broadcast_to(self.inflow_norm(inflow)[:, :, None, None], shape + [1])
        c = tf.broadcast_to(self.coords, shape + [3])
        p = tf.broadcast_to(self.par_norm(pars)[:, None, None, :], shape + [pars.shape[-1]])

        v = self.lift(tf.concat([a, c, p], -1))
        for w_t, w_x, mix in zip(self.w_t, self.w_x, self.mix):
            v = v + mix(self.spectral_conv(v, w_t, 1) + self.spectral_conv(v, w_x, 2))
        out = self.proj(v)

        if self.inlet:
            q_in = ((inflow - self.out_mean[0]) / self.out_std[0])[:, :, None]
            q = q_in * (1. - self.x) + out[..., 0] * self.x
            out = tf.stack([q, out[..., 1]], -1)
        return self.to_physical(out) if physical else out

    def pi_loss(self, inflow, pars, smooth=True):
        return super().pi_loss(inflow, pars, smooth)
