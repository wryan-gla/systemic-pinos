"""1D blood-flow physics shared by all three operators.

Flow q and pressure p in a compliant vessel satisfy the 1D mass and momentum
equations

    dA/dt + dq/dx = 0
    dq/dt + d(q^2/A)/dx + (A/rho) dp/dx + 2 nu q / (R delta) = 0

with cross-sectional area A given by a nonlinear tube law A(p). Everything here
is in the scaled units of the paper's solver: flow and time are scaled by Q0,
pressure by P0, lengths and radii are in cm.

Per-sample vessel quantities (length, radius, period, stiffness) are read out
of each model's parameter vector using a `par_idx` dict, e.g.

    par_idx = {"length": 0, "radius": 3, "period": 6, "stiffness": 7}

so any parameter layout can be used.
"""
import numpy as np
import tensorflow as tf

MU = 0.000032                 # viscosity
RHO = 0.1057                  # density
NU = MU / RHO
P0 = 10000 * 10 / 1333.22     # pressure scale
Q0 = 100.                     # flow / time scale
F1, F2 = 2e7, -35.            # tube-law stiffness: Eh/r0 = F1 exp(F2 r0) + k3
BASE_P = 57. / P0             # reference (unstressed) pressure, scaled


def vessel_pars(pars, par_idx):
    """Pick out per-sample vessel quantities, each shaped [batch, 1, 1]."""
    return {k: pars[:, i:i + 1][:, :, None] for k, i in par_idx.items()}


def tapered_radius(x, radius, length):
    """Exponentially tapering reference radius, from 1.8*radius at x=0 to radius at x=1."""
    k1 = 1.8 * radius / 1.4
    k3 = 0.4 * k1
    k2 = -tf.math.log((radius - k3) / k1) / length
    return k1 * tf.exp(-x * length * k2) + k3


def area(p, r0, stiffness):
    """Tube law: cross-sectional area from pressure."""
    ehr0 = 4. / 3. * (F1 / 1e5 * tf.exp(F2 * r0) + stiffness / 1e5)
    return ((p - BASE_P) / ehr0 + 1.) ** 2 * np.pi * r0 ** 2


def residuals(q, A, period, dA_dt, dq_dt, dq_dx, dq2A_dx, dp_dx):
    """Mass and momentum residuals given the (physical) derivatives."""
    delta = tf.sqrt(NU * period * Q0 / (2 * np.pi))  # boundary-layer thickness
    R = tf.sqrt(A / np.pi)
    mass = dA_dt + dq_dx
    momentum = dq_dt + dq2A_dx + A / RHO * dp_dx + 2 * NU * q / (R * delta)
    return mass, momentum


def _d_dt(u, dt):
    # central difference, periodic in time (axis 1)
    return (tf.roll(u, -1, axis=1) - tf.roll(u, 1, axis=1)) / (2 * dt)


def _d_dx(u, dx):
    # central difference along the vessel (axis 2), one-sided at the ends
    first = (u[:, :, 1:2] - u[:, :, :1]) / dx
    interior = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)
    last = (u[:, :, -1:] - u[:, :, -2:-1]) / dx
    return tf.concat([first, interior, last], axis=2)


def _smooth(u, size=9, sigma=3.):
    # Gaussian blur of a [batch, n_t, n_x] field
    k = tf.exp(-0.5 * ((tf.range(size, dtype=u.dtype) - size // 2) / sigma) ** 2)
    k = k[:, None] * k[None, :]
    k = (k / tf.reduce_sum(k))[:, :, None, None]
    return tf.nn.conv2d(u[..., None], k, strides=1, padding="SAME")[..., 0]


def grid_pi_loss(q, p, vp, tapering=False, smooth=False):
    """PDE residual loss on a uniform (t, x) grid using finite differences.

    q, p: physical flow and pressure, [batch, n_t, n_x], with t in [0, 1)
          (one period) and x in [0, 1] (inlet to outlet).
    vp:   output of `vessel_pars`.
    """
    n_t, n_x = q.shape[1], q.shape[2]
    if tapering:
        r0 = tapered_radius(tf.linspace(0., 1., n_x)[None, None, :], vp["radius"], vp["length"])
    else:
        r0 = vp["radius"]
    A = area(p, r0, vp["stiffness"])
    dt = vp["period"] * Q0 / n_t
    dx = vp["length"] / (n_x - 1)
    mass, momentum = residuals(
        q, A, vp["period"],
        dA_dt=_d_dt(A, dt), dq_dt=_d_dt(q, dt),
        dq_dx=_d_dx(q, dx), dq2A_dx=_d_dx(q ** 2 / A, dx), dp_dx=_d_dx(p, dx))
    if smooth:
        mass, momentum = _smooth(mass), _smooth(momentum)
    # the vessel end points are left to the boundary / junction conditions
    return tf.reduce_mean(mass[:, :, 1:-1] ** 2) + tf.reduce_mean(momentum[:, :, 1:-1] ** 2)


def bifurcation_loss(parent, daughter, sister):
    """Flow conservation and pressure continuity at a junction.

    Each argument is a (q, p) pair at the junction, shaped [batch, n_t]: the
    parent at its outlet, the daughter and its sister at their inlets.
    """
    (q_par, p_par), (q_d, p_d), (q_s, _) = parent, daughter, sister
    return tf.reduce_mean((q_par - q_d - q_s) ** 2) + tf.reduce_mean((p_par - p_d) ** 2)
