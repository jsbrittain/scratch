import pybamm
import numpy as np
import scipy
from functools import cache
from itertools import repeat
from functools import partial
import jax
import logging
from jax import lax
from jax import numpy as jnp
from jax.interpreters import ad
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters import batching
from jax.interpreters.mlir import custom_call
from jax._src.lib.mlir.dialects import hlo

from jax.lib import xla_client
import importlib.util

from functools import lru_cache

cpu_ops_spec = importlib.util.find_spec("idaklu_jax.cpu_ops")
if cpu_ops_spec:
    cpu_ops = importlib.util.module_from_spec(cpu_ops_spec)
    if cpu_ops_spec.loader:
        cpu_ops_spec.loader.exec_module(cpu_ops)

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")


num_inputs = 2
if num_inputs == 0:
    inputs = {}
elif num_inputs == 1:
    inputs = {
        "Current function [A]": 0.222,
    }
elif num_inputs == 2:
    inputs = {
        "Current function [A]": 0.222,
        "Separator porosity": 0.3,
    }
inputs0 = inputs

model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry
param = model.default_parameter_values
param.update({key: "[input]" for key in inputs.keys()})
param.process_geometry(geometry)
param.process_model(model)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
t_eval = np.linspace(0, 360, 10)
# solver = pybamm.CasadiSolver(mode="fast", rtol=1e-6, atol=1e-6)
idaklu_solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
solver = idaklu_solver

# Create surrogate data (using standard solver)
sim = idaklu_solver.solve(
    # model, t_eval=, inputs=, initial_conditions=, nproc=, calculate_sensitivities=
    model,
    t_eval,
    inputs=inputs,
    calculate_sensitivities=True,
)
data = sim["Terminal voltage [V]"](t_eval)

# Get jax expression for IDAKLU solver
output_variables = [
    "Terminal voltage [V]",
    "Discharge capacity [A.h]",
]

#######################################################################################

calculate_sensitivities = True


def isolate_var(fcn, varname):
    index = output_variables.index(varname)

    def impl(t, *args):
        out = jnp.array(fcn(t, *args))
        # Vectors always revert to (N,), meaning we don't know if they are multiple
        # time points for a single variable or multiple variables at a single time point
        if t.ndim == 0:
            t = jnp.array([t])
        if out.ndim == 1 and len(t) == 1:
            # multiple variables at a single time point
            out = out.reshape((1, out.shape[0]))
            ...
        elif out.ndim == 1 and len(t) > 1:
            # one variable at multiple time points
            out = out.reshape((len(t), 1))
        else:
            # multiple variables at multiple time points
            ...
        out = out[:, index]
        if out.shape == (1,):
            out = out[0]  # scalar
        return out

    return impl


def f(t, inputs, **kwargs):
    values = jnp.array(list(inputs.values()))
    out = f_p.bind(t, values, **kwargs)
    return out


f_p = jax.core.Primitive("f")
# f_p.multiple_results = True  # return a vector (of time samples)


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


@lru_cache(maxsize=1)  # cache the current solve for reuse
def cached_solve(model, t_eval, *args, **kwargs):
    # reconstruct time vector
    t_eval = jnp.array(list(t_eval.values()))
    return solver.solve(model, t_eval, *args, **kwargs)


@f_p.def_impl
def f_impl(t, input_values, *, invar=None):
    # convert scalar time values to vector
    if t.ndim == 0:
        t = jnp.array([t], dtype=jnp.float64)
    # reconstruct input dictionary (also make hashable for caching the solve)
    d = hashabledict()
    for ix, key in enumerate(inputs0.keys()):
        d[key] = input_values[ix].item()
    # make time vector hashable (for caching the solve)
    t_dict = hashabledict()
    for ix, tval in enumerate(t_eval):
        t_dict[ix] = tval.item()
    # Debug
    if False:
        print("Solver:")
        print("  t: ", t, type(t))
        print("  inputs: ", d)
    # Solve (or retrieve from cache)
    sim = cached_solve(
        model,
        t_dict,
        inputs=d,
        calculate_sensitivities=True,
    )
    # Primals
    out = jnp.array([jnp.array(sim[var](t)) for var in output_variables]).transpose()
    if invar is not None:
        # Return sensitivities
        tk = [k for k, tval in enumerate(t_eval) if tval >= t[0] and tval <= t[-1]]
        out = jnp.reshape(
            jnp.array(
                [
                    jnp.array(sim[outvar].sensitivities[invar][tk])
                    for outvar in output_variables
                ]
            ).transpose(),
            out.shape,
        )

    return out


def f_batch(args, batch_axes, **kwargs):
    tvec = args[0]
    if tvec.ndim == 0:
        tvec = jnp.array([tvec])
    out = jnp.array(
        list(map(lambda t: f_p.bind(t, *args[1:], **kwargs), tvec))
    ).squeeze()
    return out, batch_axes[0]


batching.primitive_batchers[f_p] = f_batch


def f_jvp(primals, tangents):
    invar = "Current function [A]"
    y = f_p.bind(*primals, invar=None)
    y_dot = f_p.bind(*primals, invar=invar)
    return y, y_dot


ad.primitive_jvps[f_p] = f_jvp


def f_transpose(y_bar, *args):
    print("f_transpose")
    primals = args[: len(args) // 2]
    tangents = args[len(args) // 2 :]  # noqa: F841
    invar = "Current function [A]"
    x_bar = f_p.bind(*primals, invar=invar)
    primals_out = (None,) * len(primals)
    tangents_out = jnp.dot(y_bar, x_bar), *((None,) * (len(primals) - 1))
    return *primals_out, *tangents_out


ad.primitive_transposes[f_p] = f_transpose


k = 5
print("\nf (scalar)")
out = f(t_eval[k], inputs)
print(out.shape, " ", out)

print("\nf (vector)")
out = f(t_eval, inputs)
print(out.shape, " ", out)

print("\nf (vmap)")
out = jax.vmap(f, in_axes=(0, None))(t_eval, inputs)
print(out.shape, " ", out)

print("\nget_var (scalar)")
outvar = output_variables[0]
out = isolate_var(f, outvar)(t_eval[k], inputs)
print(out.shape, " ", out)

print("\nget_var (vector)")
outvar = output_variables[0]
out = isolate_var(f, outvar)(t_eval, inputs)
print(out.shape, " ", out)

print("\nget_var (vmap)")
outvar = output_variables[0]
out = jax.vmap(isolate_var(f, outvar), in_axes=(0, None))(t_eval, inputs)
print(out.shape, " ", out)

print("\njvp (scalar)")
out = jax.jvp(f, (t_eval[k], inputs), (1.0, 0.0))
print(out)

print("\njacfwd (scalar)")  # SAME VALUES FOR EACH INPUT
outvar = output_variables[0]
out = jax.jacfwd(isolate_var(f, outvar), argnums=1)(t_eval[k], inputs)
print(out)

print("\njacfwd (vmap)")  # SAME VALUES FOR EACH INPUT
outvar = output_variables[0]
out = jax.vmap(jax.jacfwd(isolate_var(f, outvar), argnums=1), in_axes=(0, None))(
    t_eval, inputs
)
print(out)

print("\njacrev (scalar)")  # ALL ZEROS
outvar = output_variables[0]
out = jax.jacrev(isolate_var(f, outvar), argnums=1)(t_eval[k], inputs)
print(out)

print("\njacrev (vmap)")  # ALL ZEROS
outvar = output_variables[0]
out = jax.vmap(jax.jacrev(isolate_var(f, outvar), argnums=1), in_axes=(0, None))(
    t_eval, inputs
)
print(out)

print("\ngrad (scalar)")  # ALL ZEROS
outvar = output_variables[0]
out = jax.grad(isolate_var(f, outvar), argnums=1)(t_eval[k], inputs)
print(out)

print("\ngrad (vmap)")  # ALL ZEROS
outvar = output_variables[0]
out = jax.vmap(jax.grad(isolate_var(f, outvar), argnums=1), in_axes=(0, None))(
    t_eval, inputs
)
print(out)

# Show actual sensitivities
print("\nActual sensitivities")
outvar = output_variables[0]
check = jnp.array(
    [jnp.array(sim[outvar].sensitivities[invar]) for invar in inputs]
).squeeze()
print(check)
# assert np.allclose(out, check)
