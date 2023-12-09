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


# @jax.custom_jvp
def f(*args):
    print("f")
    print("  args: ", args)
    # Check for tracer evaluations as derivatives need to be handled differently
    flatargs, treedef = tree_flatten(args)
    if (
        # Look for BatchTracers in the inputs dictionary
        any(
            [
                isinstance(arg, jax._src.interpreters.batching.BatchTrace)
                for arg in flatargs[1:]
            ]
        )
        and False
    ):
        print("Derivative evaluation (wrt inputs)")
        # Derivative evaluation
        #
        # BatchTracer's must be divided so that pytrees return properly from
        # (purely numeric) primitive evaluations.

        # Rather than construct the appropriate structure by hand (using an evolving
        # set of JAX APIs), we mock a jax-native call and substitute our primals and
        # tangents into the returned structure.
        #
        # HOWEVER: this is not a general solution as it ignores further transformations,
        #          such as vmap, that may be applied to the function.
        n_out = len(output_variables)

        def f_mock(t, inputs):
            # Mock must depend on at least one input (and need at least one to deriv)
            key = list(inputs.keys())[0]
            return jnp.array([inputs[key], *([0.0] * (n_out - 1))], dtype="float64")

        out_mock = jax.jvp(f_mock, args, args)
        # Evaluate each requested derivative
        trace_locs = [
            ix
            for ix, val in enumerate(
                isinstance(arg, jax.core.Tracer) for arg in flatargs
            )
            if val
        ]
        primals, tangents = None, []
        for ix, arg in enumerate(flatargs):
            if not isinstance(arg, jax.core.Tracer):
                continue
            eval_flatargs = flatargs.copy()
            for ix, loc in enumerate(trace_locs):
                eval_flatargs[loc] = arg.tangent.val[ix]
            input_eval = f_p.bind(*flatargs)
            primals = input_eval.primal if primals is None else primals
            tangents.append(input_eval.tangent)
        tangents = np.array(tangents).reshape((len(trace_locs), n_out))
        # Substitute primals and tangents into mock output and return
        out = out_mock[1]
        out.primal = primals
        out.tangent.val = tangents
        return out

    print("Function evaluation")
    # Function evaluation
    return f_p.bind(*flatargs)


@f_p.def_impl
def f_impl(t, *inputs):
    if isinstance(inputs, dict):
        k1 = "in1"
        k2 = "in2"
    else:
        k1 = 0
        k2 = 1
    return np.array(
        [
            t + 7.0 * inputs[k1],
            t * t + 3.0 * inputs[k2],
            t * t * t + inputs[k1] * inputs[k2],
        ],
        dtype="float64",
    )


def f_abstract_eval(t, inputs):
    print("f_abstract_eval")
    return jax.core.ShapedArray((n_out,), t.dtype)


f_p.def_abstract_eval(f_abstract_eval)


def f_batch(args, batch_axes):
    print("f_batch")
    print("  args: ", args)
    print("  batch_axes: ", batch_axes)
    t = args[0]
    inputs = args[1:]
    out = np.array(list(map(lambda x: f(x, inputs), t)))
    return out, batch_axes[0]


batching.primitive_batchers[f_p] = f_batch


f_jvp_p = jax.core.Primitive("f_jvp")


@f_jvp_p.def_impl
def f_jvp_impl(*args):
    print("f_jvp_impl")
    print("  args: ", args)
    return np.random.rand(3)


def f_jvp(values, tangents):
    print("f_jvp")
    print("  values: ", values)
    print("  tangents: ", tangents)
    return np.array([1.0, 2.0, 3.0]), f_jvp_p.bind(*values, *tangents)


ad.primitive_jvps[f_p] = f_jvp


def f_jvp_batch(args, batch_axes):
    print("f_jvp_batch")
    print("  args: ", args)
    print("  batch_axes: ", batch_axes)
    primals = args[: len(args) // 2]
    tangents = args[len(args) // 2 :]
    t = primals[0]
    inputs = primals[1:]

    # Ensure batching occurs either over time, or over some combination of inputs
    assert (
        (batch_axes[0] is not None) and all([b is None for b in batch_axes[1:]])
    ) or ((batch_axes[0] is None) and any([b is None for b in batch_axes[1:]])), (
        "Batching must occur either over time, or over some combination of inputs, "
        "but not both or neither"
    )

    # Batch over inputs
    if any([b is not None for b in batch_axes[len(args) // 2 + 1 :]]):
        print("Batch over inputs")
        print("  batch_axes: ", batch_axes)
        print("  primals: ", primals)
        print("  tangents: ", tangents)
        # Loop over axes to batch over (index includes t to align indixes with args)
        out = []
        for ix, b in enumerate(batch_axes[len(args) // 2 :]):
            if b is None:
                continue
            input_args = primals
            val = f_jvp_impl(*input_args)
            out.append(val)
        if len(out) == 1:
            out = out[0]
        else:
            out = np.array(out)
        print("f_jvp_batch out: ", out)
        batch_axis = 0
        return out, batch_axis
        # raise NotImplementedError

    # Batch over time
    if t.ndim == 0:  # Scalar time (can occur with, e.g. jacfwd calls)
        return f_jvp_impl(*primals), batch_axes[0]
    out = np.array(list(map(lambda x: f_jvp(x, *inputs, *tangents), t)))
    batch_axis = 0  # Time is always batched over axis 0
    return out, batch_axis


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


print("\njacfwd (wrt t)")
flatin, _ = tree_flatten((t_eval[tk], inputs))
out = jax.jacfwd(f, argnums=0)(*flatin)
print(out)
assert out.shape == (n_out,)

print("\njacfwd (wrt inputs)")
out = jax.jacfwd(f, argnums=1)(t_eval[tk], inputs)
print(out)

print("\njacfwd (wrt inputs) vmap")
out = jax.vmap(jax.jacfwd(f, argnums=1), in_axes=(0, None))(t_eval, inputs)
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
