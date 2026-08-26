"""Load a trained network from ``params.pkl`` and expose it as a plain,
differentiable function of physical (x, y).

The point of this module is to hand you a callable

    f(xy) -> {"u": ..., "v": ..., "h": ..., "s": ..., "mu": ..., "C": ...}

that takes SI coordinates and returns SI fields, so that *you never touch the
normalization yourself*. Because the de-normalization is folded into the
function, ``jax.jacfwd``/``jax.grad`` of it gives derivatives in SI units
directly -- du/dx in (velocity unit)/m -- with the chain rule doing all the
scale bookkeeping. That is what makes strain rates a two-line computation
(see ``strain_rate_fn``).

Do NOT use ``solver.state.solution[1]`` for this. That built-in gradient
function returns *normalized* gradients pre-multiplied by an internal
``coeff`` (see the "[IMPORTANT NOTE]" comment in
``diffice_jax/model/xpinns/networks.py``), which is the convention the
equation residuals need, not SI.

Usage as a library
------------------
    from examples.load_trained_network import load_trained_network

    net = load_trained_network("path/to/config.yaml", "path/to/solver_dir")

    # xy: (N, 2) array of x, y in metres, in the dataset's own projection.
    e = net.strain_rate(xy, per_year=True)
    e["exx"], e["eyy"], e["exy"], e["e_eff"]      # 1/yr

    fields = net.field(xy)           # u, v [m/s], h, s [m], mu [Pa s], C
    jac = net.jacobian(xy)           # every field's d/dx and d/dy, e.g. jac["h_x"]

Everything is SI unless you ask otherwise: velocities are m/s (~1e-5, not
~100) and strain rates 1/s. Use ``per_year=True`` or ``SECONDS_PER_YEAR``.

Usage as a script (sanity check / demo)
---------------------------------------
    python examples/load_trained_network.py <config.yaml> <solver_dir> [--region 0]

Caveats
-------
* ``prepare()`` needs the training dataset on disk (it recomputes the
  normalization from it), and the yaml must match the one used for training in
  its structural parts (workflow, network depth/width, region count, data
  source). Nothing validates the pytree shape -- a mismatch surfaces as a
  cryptic shape error on the first evaluation.
* For an XPINN, each sub-region has its own network valid only inside its own
  sub-region. Evaluating region 0's network at region 3's coordinates returns
  confident nonsense. Pass ``region=`` to pick the right one.
* Nothing clamps (x, y) to the trained footprint; extrapolation is unbounded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

import diffice_jax as djax


# Column layout of the raw network output, by workflow. mu/C come out already
# exponentiated (jnp.exp inside solu_create), so they are positive and only
# need multiplying by their scale.
#   XPINN, any region:  u, v, h, s, mu, C
#   PINN grounded:      u, v, h, s, mu, C
#   PINN floating:      u, v, h, mu          <- no s, and mu is column 3
_COLS_WITH_S = ("u", "v", "h", "s", "mu", "C")
_COLS_NO_S = ("u", "v", "h", "mu")

# The DIFFICE datasets are in SI, so velocities come out in m/s (~1e-5, not
# ~100) and strain rates in 1/s (~1e-10). Multiply by this to get m/yr and
# 1/yr, which is what you almost certainly want to plot. Same constant the
# solver stamps into field_and_residual_bundle().
SECONDS_PER_YEAR = 31557600.0


@dataclass
class TrainedNetwork:
    """A loaded solver plus SI-unit callables for one sub-region."""

    solver: object
    region: int
    field: Callable          # (N, 2) -> dict of (N, 1) SI fields
    jacobian: Callable       # (N, 2) -> dict of (N, 1) SI d<field>/dx, d<field>/dy
    strain_rate: Callable    # (N, 2) -> dict with exx, eyy, exy, e_eff
    field_point: Callable    # (2,) -> (ncol,) SI vector; differentiate this yourself
    columns: tuple           # names of field_point's output columns, in order


def load_trained_network(config_path, solver_dir, region=0, params_name="params.pkl"):
    """Build the solver, load ``params.pkl``, and wrap it as a TrainedNetwork.

    :param config_path: the workflow yaml used for training
    :param solver_dir: directory containing params.pkl (and config.json)
    :param region: sub-region position for an XPINN; ignored for a PINN
    """

    solver_dir = Path(solver_dir)
    config = djax.load_workflow_config(config_path)
    solver = djax.build_solver_from_config(config).prepare()
    solver.load_params(solver_dir / params_name)
    # _warn_on_config_drift(solver, solver_dir)
    return wrap_solver(solver, region=region)


def wrap_solver(solver, region=0):
    """Wrap an already-prepared, params-loaded solver. Split out from
    ``load_trained_network`` so you can reuse a solver you already have."""

    state = solver.state
    if state is None:
        raise RuntimeError("Call prepare() (and load_params) on the solver first.")

    xpinn = solver.model_config.workflow == "xpinn"
    if xpinn:
        n_sub = len(state.sub_region_indices)
        if not 0 <= region < n_sub:
            raise IndexError(f"region {region} out of range; solver has {n_sub} sub-region(s)")
        # solu_create indexes params['net_u'][idx] / basal_mask[idx] / scale[idx]
        # with this same idx, so it is the position in the sub-network list.
        idx = int(state.sub_region_indices[region])
        raw_net = lambda X: state.solution[0](state.params, X, idx)
    else:
        region = 0
        raw_net = lambda X: state.solution(state.params, X)

    scale = state.scales[region]
    dmean, drange, dyn = scale.data_mean, scale.data_range, scale.dynamic_scale
    basal = bool(state.basal_mask[region])
    has_s = xpinn or basal
    columns = _COLS_WITH_S if has_s else _COLS_NO_S

    def field_point(xy):
        """(2,) physical x, y -> (ncol,) SI field vector. Differentiable."""
        xy = jnp.asarray(xy).reshape(1, 2)
        X = jnp.hstack([(xy[:, 0:1] - dmean.x_mean) / drange.x_range,
                        (xy[:, 1:2] - dmean.y_mean) / drange.y_range])
        out = raw_net(X)[0]
        mu_col = 4 if has_s else 3
        vals = [out[0] * drange.u_range + dmean.u_mean,
                out[1] * drange.v_range + dmean.v_mean,
                # h carries no offset -- it is normalized by its mean alone.
                out[2] * dmean.h_mean]
        if has_s:
            # XPINN normalizes s by (s_range, s_mean); a grounded PINN by h_mean.
            vals.append(out[3] * drange.s_range + dmean.s_mean if xpinn
                        else out[3] * dmean.h_mean)
        vals.append(out[mu_col] * dyn.mu0)
        if has_s:
            # C is structurally zero (not NaN) on a floating XPINN region
            # unless the run was forward-mode / all_regions_c.
            vals.append(out[5] * dyn.c0 if out.shape[0] > 5 else jnp.nan)
        return jnp.stack(vals)

    field_batch = jax.jit(jax.vmap(field_point))
    jac_batch = jax.jit(jax.vmap(jax.jacfwd(field_point)))

    def field(xy):
        out = field_batch(_as_points(xy))
        return {name: out[:, i:i + 1] for i, name in enumerate(columns)}

    def jacobian(xy):
        jac = jac_batch(_as_points(xy))          # (N, ncol, 2)
        out = {}
        for i, name in enumerate(columns):
            out[f"{name}_x"] = jac[:, i, 0:1]
            out[f"{name}_y"] = jac[:, i, 1:2]
        return out

    def strain_rate(xy, per_year=False):
        """SSA strain-rate components, in 1/s (SI, matching the dataset) or in
        1/yr with ``per_year=True``. Also returns the effective strain rate
        used by DIFFICE's viscosity relation."""
        jac = jac_batch(_as_points(xy))
        u_x, u_y = jac[:, 0, 0:1], jac[:, 0, 1:2]
        v_x, v_y = jac[:, 1, 0:1], jac[:, 1, 1:2]
        exy = 0.5 * (u_y + v_x)
        # Matches the strate expression in solu_create's gradf: incompressible
        # SSA effective strain rate, with e_zz = -(exx + eyy) folded in.
        e_eff = jnp.sqrt(u_x ** 2 + v_y ** 2 + exy ** 2 + u_x * v_y)
        out = {"u_x": u_x, "u_y": u_y, "v_x": v_x, "v_y": v_y,
               "exx": u_x, "eyy": v_y, "exy": exy, "e_eff": e_eff}
        return {k: v * SECONDS_PER_YEAR for k, v in out.items()} if per_year else out

    return TrainedNetwork(solver=solver, region=region, field=field,
                          jacobian=jacobian, strain_rate=strain_rate,
                          field_point=field_point, columns=columns)


def _as_points(xy):
    xy = jnp.asarray(xy, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    if xy.ndim == 1:
        xy = xy.reshape(1, 2)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"expected (N, 2) coordinates, got shape {tuple(xy.shape)}")
    return xy


def _warn_on_config_drift(solver, solver_dir):
    """config.json is solver.save()'s record of what actually produced
    params.pkl. It is a resolved dataclass dump, not a reloadable yaml, so we
    only spot-check the two fields most likely to be wrong."""

    path = solver_dir / "config.json"
    if not path.exists():
        print(f"[warn] no config.json in {solver_dir}; cannot verify the yaml matches params.pkl")
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("model", {}).get("workflow") not in (None, solver.model_config.workflow):
        print(f"[warn] model.workflow differs: yaml={solver.model_config.workflow!r} "
              f"but training used {saved['model']['workflow']!r}")
    want_src = saved.get("data", {}).get("source")
    if want_src is not None:
        # Compare resolved paths -- the saved config may be absolute where the
        # yaml is relative to its own directory, which is not a real mismatch.
        got = Path(solver.data_config.source).resolve()
        want = Path(want_src).resolve()
        if got != want:
            print(f"[warn] data.source differs: yaml={got} but training used {want}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="workflow yaml used for training")
    ap.add_argument("solver_dir", help="directory containing params.pkl")
    ap.add_argument("--region", type=int, default=0, help="XPINN sub-region position")
    ap.add_argument("--params-name", default="params.pkl")
    args = ap.parse_args()

    net = load_trained_network(args.config, args.solver_dir,
                              region=args.region, params_name=args.params_name)

    # Sanity demo: evaluate at some of the region's own data points, where the
    # network is actually valid, rather than at made-up coordinates. Spread the
    # sample across the point list -- the first few rows sit on the region edge
    # where the velocity is ~0 and the numbers look degenerate.
    pred = net.solver.predict()
    r = pred["regions"][args.region]
    all_xy = jnp.hstack([jnp.asarray(r["x"]), jnp.asarray(r["y"])])
    take = jnp.linspace(0, all_xy.shape[0] - 1, 5).astype(int)
    xy = all_xy[take]

    print(f"workflow={net.solver.model_config.workflow} region={net.region} "
          f"kind={r['kind']} columns={net.columns} npoints={all_xy.shape[0]}")
    f = net.field(xy)
    e = net.strain_rate(xy, per_year=True)
    print(f"\n{'x [m]':>12} {'y [m]':>12} {'u [m/yr]':>10} {'v [m/yr]':>10} "
          f"{'mu [Pa s]':>11} {'exx [1/yr]':>11} {'eyy [1/yr]':>11} "
          f"{'exy [1/yr]':>11} {'e_eff[1/yr]':>11}")
    for i in range(xy.shape[0]):
        print(f"{xy[i,0]:12.1f} {xy[i,1]:12.1f} "
              f"{f['u'][i,0]*SECONDS_PER_YEAR:10.2f} {f['v'][i,0]*SECONDS_PER_YEAR:10.2f} "
              f"{f['mu'][i,0]:11.3e} {e['exx'][i,0]:11.3e} {e['eyy'][i,0]:11.3e} "
              f"{e['exy'][i,0]:11.3e} {e['e_eff'][i,0]:11.3e}")

    # Cross-check the wrapper against solver.predict(), which does the same
    # de-normalization independently. Report relative error -- these run in
    # float32 by default, so expect ~1e-6, not exact equality.
    def rel(name):
        ref = jnp.asarray(r[name])[take]
        scale = jnp.maximum(jnp.max(jnp.abs(ref)), 1e-30)
        return float(jnp.max(jnp.abs(f[name] - ref)) / scale)

    checks = "  ".join(f"{n}={rel(n):.2e}" for n in ("u", "v", "h", "mu") if n in f)
    print(f"\nmax relative disagreement with solver.predict():  {checks}")


if __name__ == "__main__":
    main()
