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

from jax.tree_util import tree_flatten, tree_unflatten

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
    "Total lithium [mol]",
]

#######################################################################################


def isolate_var(f, varname):
    index = varnames.index(varname)

    def f_isolated(*args, **kwargs):
        return f(*args, **kwargs)[index]

    return f_isolated


f_p = jax.core.Primitive("f")


def f(*args):
    print("f")
    flatargs, treedef = tree_flatten(args)
    out = f_p.bind(*flatargs)
    print("<- f: ", out)
    return out


@f_p.def_impl
def f_impl(t, *inputs):
    print("f_impl")
    if isinstance(inputs, dict):
        k1 = "in1"
        k2 = "in2"
    else:
        k1 = 0
        k2 = 1
    out = np.array(
        [
            t + 7.0 * inputs[k1],
            t * t + 3.0 * inputs[k2],
            t * t * t + inputs[k1] * inputs[k2],
        ],
        dtype="float64",
    )
    print("<- f_impl: ", out)
    return out


# def f_abstract_eval(t, inputs):
#    print('f_abstract_eval')
#    return jax.core.ShapedArray((n_out,), t.dtype)


# f_p.def_abstract_eval(f_abstract_eval)


def f_batch(args, batch_axes):
    print("f_batch")
    print("  args: ", args)
    print("  batch_axes: ", batch_axes)
    t = args[0]
    inputs = args[1:]
    out = np.array(list(map(lambda x: f(x, inputs), t)))
    return out, batch_axes[0]


batching.primitive_batchers[f_p] = f_batch


def f_jvp(primals, tangents):
    print("f_jvp")
    print("  values: ", primals)
    print("  tangents: ", tangents)

    def make_zero(prim, tan):
        return lax.zeros_like_array(prim) if type(tan) is ad.Zero else tan

    zero_mapped_tangents = tuple(
        map(lambda pt: make_zero(pt[0], pt[1]), zip(primals, tangents))
    )
    y = f(*primals)
    y_dot = f_jvp_p.bind(*primals, *zero_mapped_tangents)
    print("  y: ", y)
    print("  y_dot: ", y_dot)
    print("<- f_jvp")
    return y, y_dot


ad.primitive_jvps[f_p] = f_jvp


f_jvp_p = jax.core.Primitive("f_jvp")


@f_jvp_p.def_impl
def f_jvp_impl(*args):
    print("f_jvp_impl")
    return np.random.rand(3)


def f_jvp_abstract_eval(*args):
    print("f_jvp_abstract_eval")
    print("  args: ", args)
    primals = args[: len(args) // 2]
    tangents = args[len(args) // 2 :]
    t = primals[0]
    out = jax.core.ShapedArray((n_out,), t.dtype)
    print("<- f_jvp_abstract_eval")
    return out


f_jvp_p.def_abstract_eval(f_jvp_abstract_eval)


f_vjp_p = jax.core.Primitive("f_vjp")


def f_jvp_transpose(y_bar, *args):
    print("f_jvp_transpose: ")
    print("  y_bar: ", y_bar)
    print("  args: ", args)

    primals = args[: len(args) // 2]
    tangents = args[len(args) // 2 :]

    # assert isinstance(tangents[0], jax.Array), \
    #    "Tranpose rule can only be applied to input variables, not time"

    tangents_out = []
    for _ in range(len(tangents)):
        tangents_out.append(f_vjp_p.bind(y_bar, *primals))
    print("  tangents_out: ", tangents_out)
    print(tangents_out)

    out = *([None] * len(tangents_out)), *tangents_out
    print("<- f_jvp_transpose")
    return out


ad.primitive_transposes[f_jvp_p] = f_jvp_transpose


def f_vjp(y_bar, primals):
    print("f_vjp")
    print("  y_bar: ", y_bar)
    print("  primals: ", primals)
    return f_vjp_p.bind(y_bar, primals)


@f_vjp_p.def_impl
def f_vjp_p_impl(y_bar, *primals):
    print("f_vjp_p_impl: ")
    print("  y_bar: ", y_bar)
    return np.random.rand(3)


def f_vjp_p_batch(args, batch_axes):
    print("f_vjp_p_batch")
    print("  args: ", args)
    print("  batch_axes: ", batch_axes)
    y_bars, t, *inputs = args

    # out = []
    # for y_bar in y_bars:
    #    out.append(f_vjp(y_bar, (t, *inputs)))

    out = []
    if t.ndim > 0:
        for t in t:
            out.append(f_vjp(y_bars, (t, *inputs)))
    else:
        out.append(f_vjp(y_bars, (t, *inputs)))

    out = np.array(out)
    batch_axis = 0
    print("<- f_vjp_p_batch: ", out)
    return out, batch_axis


batching.primitive_batchers[f_vjp_p] = f_vjp_p_batch


def f_jvp_batch(args, batch_axes):
    print("f_jvp_batch")
    print("  args: ", args)
    print("  batch_axes: ", batch_axes)
    primals = args[: len(args) // 2]
    tangents = args[len(args) // 2 :]
    t = primals[0]
    inputs = primals[1:]
    tangent_batch_axes = batch_axes[len(args) // 2 :]

    if (
        any([b is not None for b in tangent_batch_axes[1:]])
        and batch_axes[0] is None
        and tangent_batch_axes[0] is None
    ):
        # Tangent batching over some set of inputs
        out = []
        batch_indices = [
            ix for ix, b in enumerate(batch_axes[len(args) // 2 :]) if b is not None
        ]
        # Compute derivates for each requested input
        for ix, batch_index in enumerate(batch_indices):
            # Create new tangents vector
            input_tangents = list(tangents)
            replacement_values = list(tangents[batch_index])
            for ix, val in enumerate(replacement_values):
                input_tangents[batch_indices[ix]] = val
            # Compute derivatives for the given input vector
            item_primals, item_tangents = f_jvp(primals, tuple(input_tangents))
            if isinstance(item_tangents, np.ndarray):
                out.append(item_tangents)
            else:
                out.append(item_tangents.val)
        if len(out) == 1:
            # Report no batching
            return out[0], None
        # Report batching over the requested inputs
        out = np.array(out)
        # Keep axes separated for now: [time, outvars, invars]
        out = out.reshape(-1, out.shape[1], out.shape[0])  # remapping inputs to dim 2
        print("f_jvp_batch out [", out.shape, "]: ", out)
        batch_axis = 2
        return out, batch_axis
    elif batch_axes[0] is not None and all([b is None for b in batch_axes[1:]]):
        # Batching over time
        if t.ndim == 0:  # Scalar time (can occur with, e.g. jacfwd calls)
            return f_jvp(primals, tangents)[1], None
        out = np.array(list(map(lambda x: f_jvp((x, *inputs), tangents)[1], t)))
        batch_axis = 0  # Time is always batched over axis 0
        return out, batch_axis
    elif (
        # Differentiate wrt time
        tangent_batch_axes[0]
        is not None
    ):
        # raise Exception("Cannot differentiate with respect to time.")
        # Q: Can this be done wrt the 'Time [secs]' variable?

        # (just batch over time for now)
        if t.ndim == 0:  # Scalar time (can occur with, e.g. jacfwd calls)
            return f_jvp(primals, tangents)[1], None
        out = np.array(list(map(lambda x: f_jvp((x, *inputs), tangents)[1], t)))
        batch_axis = 0  # Time is always batched over axis 0
        return out, batch_axis
    else:
        raise Exception(
            f"Batching can occur over time, or differentiation over "
            "inputs, but got: batch_axes = {batch_axes}."
        )


batching.primitive_batchers[f_jvp_p] = f_jvp_batch


varnames = ["out1", "out2", "out3"]
n_in = len(inputs)
n_out = len(output_variables)
inputs = {"in1": 1.0, "in2": 2.0}
t_eval = np.arange(20, dtype="float64")
tk = 5

print("\nf")
out = f(t_eval[tk], inputs)
assert out.shape == (n_out,)
print(type(out), out)

print("\nf vmap")
out = jax.vmap(f, in_axes=(0, None))(t_eval, inputs)
assert out.shape == (len(t_eval), n_out)
print(type(out), out)

print("\njvp")
out = jax.jvp(f, (t_eval[tk], inputs), (1.0, inputs))
print(out)

# print('\njacfwd (wrt t)')
# out = jax.jacfwd(f, argnums=0)(t_eval[tk], inputs)
# print(out)
# assert out.shape == (n_out,)

# print('\njacfwd (wrt t) vmap')
# out = jax.vmap(jax.jacfwd(f, argnums=0), in_axes=(0, None))(t_eval, inputs)
# print(out)
# assert out.shape == (len(t_eval), n_out)

# print('\njacfwd (wrt inputs)')
# out = jax.jacfwd(f, argnums=1)(t_eval[tk], inputs)
# print(out)

# print('\njacfwd (wrt inputs) vmap')
# out = jax.vmap(jax.jacfwd(f, argnums=1), in_axes=(0, None))(t_eval, inputs)
# print(out)

print("\njacrev (wrt inputs)")  # Should be same output as jacfwd
out = jax.jacrev(f, argnums=1)(t_eval[tk], inputs)
print(out)

print("\ngrad (wrt inputs) vmap")
out = jax.vmap(
    jax.jacrev(
        isolate_var(f, varnames[0]),
        argnums=1,
    ),
    in_axes=(0, None),
)(t_eval, inputs)
print(out)

if False:
    print("\ngrad (wrt inputs)")
    out = jax.grad(isolate_var(f, varnames[0]), argnums=1)(t_eval[tk], inputs)
    print(out)

    print("\ngrad (wrt inputs) vmap")
    out = jax.vmap(
        jax.grad(
            isolate_var(f, varnames[0]),
            argnums=1,
        ),
        in_axes=(0, None),
    )(t_eval, inputs)
    print(out)

exit(0)

print("\njacrev (wrt inputs)")
out = jax.jacrev(f, argnums=1)(t_eval[tk], inputs)
print(out)

assert isinstance(out, dict)
print("len of output: ", len(out), type(out))
for k, o in out.items():
    print(o.shape, type(o))
    print(o)
    assert o.shape == (n_out,)

exit(0)

print("\njacfwd (wrt inputs) vmap")
out = jax.vmap(jax.jacfwd(f, argnums=1), in_axes=(0, None))(t_eval, inputs)
print(out)
assert isinstance(out, dict)
print("len of output: ", len(out), type(out))
for k, o in out.items():
    print(o.shape, type(o))
    print(o)
    assert o.shape == (len(t_eval), n_out)

print("\njacrev")  # RETURNS ALL ZEROS
out = jax.jacrev(f, argnums=1)(t_eval[tk], inputs)
assert isinstance(out, dict)
print("len of output: ", len(out), type(out))
for k, o in out.items():
    print(o.shape, type(o))
    print(o)
    assert o.shape == (n_out,)

print("\njacrev vmap")  # RETURNS ALL ZEROS
out = jax.vmap(jax.jacrev(f, argnums=1), in_axes=(0, None))(t_eval, inputs)
assert isinstance(out, dict)
print("len of output: ", len(out), type(out))
for k, o in out.items():
    print(o.shape, type(o))
    print(o)
    assert o.shape == (len(t_eval), n_out)

print("\njacfwd-jacrev")
out = jax.jacrev(jax.jacfwd(f, argnums=1), argnums=1)(t_eval[tk], inputs)
print("len of output: ", len(out), type(out))
assert isinstance(out, dict)
assert len(out) == n_in
for k, o in out.items():
    print(o)
    assert isinstance(o, dict)

print("\nisolate_var")
for varname in varnames:
    out = isolate_var(f, varname)(t_eval[tk], inputs)
    print(varname, type(out), out)
    assert out.ndim == 0

print("\nisolate_var vmap")
for varname in varnames:
    out = jax.vmap(isolate_var(f, varname), in_axes=(0, None))(t_eval, inputs)
    print(varname, type(out), out)
    assert out.shape == (len(t_eval),)

print("\ngrad")
for varname in varnames:
    out = jax.grad(isolate_var(f, varname), argnums=1)(t_eval[tk], inputs)
    print(varname)
    assert isinstance(out, dict)
    for k, o in out.items():
        print(o)
        assert o.ndim == 0

print("\ngrad vmap")
for varname in varnames:
    out = jax.vmap(jax.grad(isolate_var(f, varname), argnums=1), in_axes=(0, None))(
        t_eval, inputs
    )
    print(varname)
    assert isinstance(out, dict)
    for k, o in out.items():
        print(o)
        assert o.shape == (len(t_eval),)
