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

cpu_ops_spec = importlib.util.find_spec("idaklu_jax.cpu_ops")
cpu_ops = importlib.util.module_from_spec(cpu_ops_spec)
cpu_ops_spec.loader.exec_module(cpu_ops)

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")


# Custom logger
class logger:
    NONE = 0
    INFO = 1
    DEBUG = 2

    level = NONE
    logging_fcns = [print]  # list of logging functions, can be file outputs, etc.

    @classmethod
    def setLevel(cls, level):
        cls.level = level

    @classmethod
    def log(cls, *args, **kwargs):
        for log_out in cls.logging_fcns:
            log_out("    ", args[0])
            for arg in args[1:]:
                log_out("      ", arg)

    @classmethod
    def info(cls, *args, **kwargs):
        if cls.level >= cls.INFO:
            cls.log(*args, **kwargs)

    @classmethod
    def debug(cls, *args, **kwargs):
        if cls.level >= cls.DEBUG:
            cls.log(*args, **kwargs)


logger.setLevel(logger.NONE)

num_inputs = 1
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
solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)

data = solver.solve(
    model,
    t_eval,
    inputs=inputs,
)[
    "Terminal voltage [V]"
](t_eval)


vars_out = ["Terminal Voltage [V]"]
vars_in = ["Current function [A]"]


# IMPLEMENTATION DEFINITION
#
# See https://github.com/google/jax/blob/main/jax/_src/core.py
# for Primitive class definition


def jaxify_solve(t, *inputs):
    logger.info("jaxify_solve: ", type(t), type(inputs))
    if not isinstance(t, list) and not isinstance(t, np.ndarray):
        t = [t]
    # Reconstruct dictionary of inputs (relies on 'inputs0' being in scope)
    d = inputs0.copy()
    for ix, (k, v) in enumerate(inputs0.items()):
        d[k] = inputs[ix]
    # Solver
    sim = solver.solve(
        model,
        t_eval,
        inputs=dict(d),
        calculate_sensitivities=True,
    )
    term_v = sim["Terminal voltage [V]"]
    term_v_sens = sim["Terminal voltage [V]"].sensitivities["Current function [A]"]
    if len(t) == 1:
        tk = np.abs(t_eval - t).argmin()
        return term_v(t)[0], np.array(term_v_sens[tk])[0][0]
    else:
        return (
            term_v(t_eval),  # np.array
            np.array(term_v_sens).squeeze(),
        )  # casadi.DM -> 2d -> 1d


# JAX PRIMITIVE DEFINITION

f_p = jax.core.Primitive("f")
# f_p.multiple_results = True  # return a vector (of time samples)


def f(t, *inputs):
    """
    Takes a dictionary of inputs when called directly, e.g.:
        inputs = {
            'Current function [A]': 0.222,
            'Separator porosity': 0.3,
        }
    but can also take a vector of inputs when called by other primitive functions
    such as jvp (may need to check if we can remove that call)

    Ordering should be okay as it is determined ONLY by this function (i.e. there is no
    other dictionary unpacking in the code)
    """
    logger.info("f: ", type(t), type(inputs))
    dv = list(inputs[0].values())
    bind_point = f_p.bind(t, *dv, *inputs[1:])
    logger.debug("f [exit]: ", (type(bind_point), bind_point))
    return bind_point


@f_p.def_impl
def f_impl(*inputs):
    """Concrete implementation of Primitive"""
    logger.info("f_impl: ", type(inputs))
    term_v, term_v_sens = jaxify_solve(*inputs)
    logger.debug("f_impl [exit]: ", (type(term_v), term_v))
    return term_v


@f_p.def_abstract_eval
def f_abstract_eval(t, *inputs):
    """Abstract evaluation of Primitive
    Takes abstractions of inputs, returned ShapedArray for result of primitive
    """
    logger.info("f_abstract_eval")
    y_aval = jax.core.ShapedArray(t.shape, t.dtype)
    return y_aval


def f_batch(args, batch_axes):
    """Batching rule for Primitive
    Takes batched inputs, returns batched outputs and batched axes
    """
    logger.info("f_batch: ", type(args), type(batch_axes))
    return f_p.bind(*args), batch_axes[0]
    # return f(*args), batch_axes[0]


batching.primitive_batchers[f_p] = f_batch


# def f_lowering(ctx, mean_anom, ecc, *, platform="cpu"):
def f_lowering_cpu(ctx, t, *inputs):
    logger.info("jaxify_lowering: ")
    t_aval = ctx.avals_in[0]
    np_dtype = t_aval.dtype

    if np_dtype == np.float64:
        op_name = "cpu_kepler_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    dtype = mlir.ir.RankedTensorType(t.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))
    size = np.prod(dims).astype(np.int64)
    results = custom_call(
        op_name,
        result_types=[dtype],  # ...
        operands=[
            mlir.ir_constant(size),
            t,
            t,
        ],  # TODO: Passing t twice to simulate inputs of equal length
        operand_layouts=[(), layout, layout],
        result_layouts=[layout],  # ...
    ).results
    return results


# f_p.def_impl(partial(xla.apply_primitive, f_p))
# f_p.def_abstract_eval(f_abstract_eval)
mlir.register_lowering(
    f_p,
    f_lowering_cpu,
    platform="cpu",
)

# JVP / Forward-mode autodiff / J.v len(v)=num_inputs / len(return)=num_outputs


def f_jvp(primals, tangents):
    """JVP rule

    Given values of the arguments and perturbation of the arguments (tangents),
    compute the output of the primitive and the perturbation of the output.

    Must be JAX-traceable (may be invoked with abstract values)
    Note that we subvert this requirement by binding another primitive instead

    Args:
        primals: Tuple of primals
        tangents: Tuple of tangents; same structure as primals; some may be ad.Zero
            to specify a zero tangent;
    Returns:
        a (primals_out, tangents_out) pair where primals_out is a fun(*primals) and
        tangents_out is the Jacobian-vector product of (fun @ primals) with tangents;
        J.v; tangents_out has the same Python tree structure and shape as primals_out.
    """
    logger.info("f_jvp: ", *list(map(type, (*primals, *tangents))))

    # Deal with Zero tangents
    def make_zero(prim, tan):
        return lax.zeros_like_array(prim) if type(tan) is ad.Zero else tan

    zero_mapped_tangents = tuple(
        map(lambda pt: make_zero(pt[0], pt[1]), zip(primals, tangents))
    )
    # y = f(*primals)  # primals_out = fun(*primals); (unnecessary call?)
    y = f_p.bind(*primals)  # returns numpy array
    y_dot = f_jvp_p.bind(  # tangents_out = J.v
        *primals,
        *zero_mapped_tangents,
    )
    logger.debug("f_jvp [exit]: ", (type(y), y), (type(y_dot), y_dot))
    return y, y_dot


ad.primitive_jvps[f_p] = f_jvp
f_jvp_p = jax.core.Primitive("f_jvp")


# @f_jvp_p.def_impl
# def f_jvp_p_eval(primals, tangents):
#     logger.info("f_jvp_p_eval: ", type(primals), type(tangents))
#     x_t, x_params, x_t_eval = primals
#     y, y_dot = jaxify_solve(x_t, x_params, x_t_eval)
#     return y_dot[0]


@f_jvp_p.def_abstract_eval
def f_jvp_abstract_eval(*args):
    logger.info("f_jvp_abstract_eval: ")
    t_dot = args[len(args) // 2]
    y_dot_aval = jax.core.ShapedArray(t_dot.shape, t_dot.dtype)
    logger.debug("f_jvp_abstract_eval [exit]: ", (type(y_dot_aval), y_dot_aval))
    return y_dot_aval


def f_jvp_transpose(y_bar, *args):
    """JVP transpose rule

    Args:
        y_bar: cotangent of the output of the primitive

    """
    logger.info("f_jvp_transpose: ")
    # assert ad.is_undefined_primal(x_dot_dummy)
    primals = args[: len(args) // 2]
    tangents = args[len(args) // 2 :]  # noqa: F841
    x_bar = f_vjp_p.bind(*primals, y_bar)
    logger.debug("j_jvp_transpose [exit]: ", (type(x_bar), x_bar), (type(y_bar), y_bar))
    primals_out = (None,) * len(primals)
    tangents_out = jnp.dot(y_bar, x_bar), *(
        (None,) * (len(primals) - 1)
    )  # TODO: Generalise function
    return *primals_out, *tangents_out


ad.primitive_transposes[f_jvp_p] = f_jvp_transpose

# VJP / Reverse-mode autodiff / v^T.J len(v)=num_outputs / len(return)=num_inputs

f_vjp_p = jax.core.Primitive("f_vjp")


@f_vjp_p.def_impl
def f_vjp_impl(*args):
    y_bar = args[-1]  # noqa: F841
    args = args[:-1]
    logger.info("f_vjp_impl: ")
    term_v, term_v_sens = jaxify_solve(*args)
    logger.debug("f_vjp_impl [exit]: ", (type(term_v_sens), term_v_sens))
    return term_v_sens


@f_vjp_p.def_abstract_eval
def f_vjp_abstract_eval(t, *args):
    y_aval = jax.core.ShapedArray(t.shape, t.dtype)
    return y_aval


def f_vjp_batch(args, batch_axes):
    logger.info("f_vjp_batch: ", type(args), type(batch_axes))
    y_bar = args[-1]  # noqa: F841
    # concrete implemenatation provides native batching
    term_v, term_v_sens = jaxify_solve(*args[:-1])
    term_v_sens = np.array(term_v_sens)
    logger.debug("f_vjp_batch [exit]: ", (type(term_v_sens), term_v_sens))
    return term_v_sens, batch_axes[0]


batching.primitive_batchers[f_vjp_p] = f_vjp_batch


# def f_lowering(ctx, mean_anom, ecc, *, platform="cpu"):
def f_vjp_lowering_cpu(ctx, t, *inputs):
    # TODO: This is just a copy of the f_p lowering function for now
    logger.info("jaxify_lowering: ")
    t_aval = ctx.avals_in[0]
    np_dtype = t_aval.dtype

    if np_dtype == np.float64:
        op_name = "cpu_kepler_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    dtype = mlir.ir.RankedTensorType(t.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))
    size = np.prod(dims).astype(np.int64)
    results = custom_call(
        op_name,
        result_types=[dtype],  # ...
        operands=[
            mlir.ir_constant(size),
            t,
            t,
        ],  # TODO: Passing t twice to simulate inputs of equal length
        operand_layouts=[(), layout, layout],
        result_layouts=[layout],  # ...
    ).results
    return results


# f_p.def_impl(partial(xla.apply_primitive, f_p))
# f_p.def_abstract_eval(f_abstract_eval)
mlir.register_lowering(
    f_vjp_p,
    f_vjp_lowering_cpu,
    platform="cpu",
)


# TEST

# t_eval = np.linspace(0.0, 360, 10)
x = inputs  # np.array(list(inputs.values()))

d_axes = None  # dict(zip(inputs.keys(), repeat(None)))  # can also be dict
in_axes = (0, d_axes)

print(f"\nTesting with input: {x=}")

# Scalar evaluation

k = 5
print("\neval with scalar t:")  # calls f
print(f(t_eval[k], inputs))

print("\ngrad with scalar t:")  # calls f
print(jax.grad(f, argnums=0)(t_eval[k], x))

print("\nvalue_and_grad with scalar t:")  # calls f
value_and_grad_f = jax.value_and_grad(f)
print(value_and_grad_f(t_eval[k], x))

print("\njit eval with scalar t:")  # calls f
print(jax.jit(f)(t_eval, x))

print("\njit grad with scalar t:")  # calls f
print(jax.grad(jax.jit(f), argnums=0)(t_eval[k], x))

print("\njit value_and_grad with scalar t:")
jit_value_and_grad_f = jax.value_and_grad(jax.jit(f))  # works, but does NOT call f
print(jit_value_and_grad_f(t_eval[k], x))
jit_value_and_grad_f = jax.jit(jax.value_and_grad(f))  # calls f
print(jit_value_and_grad_f(t_eval[k], x))

# Vector evaluation

# Form input axes

print("\neval with vmap over t:")  # calls f
vmap_f = jax.vmap(f, in_axes=in_axes)
print(vmap_f(t_eval, x))

print("\ngrad with vmap over t:")  # calls f
vmap_grad = jax.vmap(jax.grad(f), in_axes=in_axes)
print(vmap_grad(t_eval, x))

print("\nvalue_and_grad with vmap over t:")  # calls f
vmap_vg = jax.vmap(jax.value_and_grad(f), in_axes=in_axes)
print(vmap_vg(t_eval, x))

print("\njit eval with vmap over t:")  # calls f
vmap_jit = jax.vmap(jax.jit(f), in_axes=in_axes)
print(vmap_jit(t_eval, x))

# TODO: These functions are calling jaxify_solve, not the lowering function
# print("\ngrad with vmap over t:")  # calls f
# vmap_grad_jit = jax.vmap(jax.grad(jax.jit(f)), in_axes=in_axes)
# print(vmap_grad_jit(t_eval, x))

# print("\njit value_and_grad with vmap over t:")  # calls f
# jit_vmap_vg = jax.vmap(jax.value_and_grad(jax.jit(f)), in_axes=in_axes)
# print(jit_vmap_vg(t_eval, x))


# Differentiate downstream expressions

# Original:
t = t_eval
term_v, term_v_sens = vmap_vg(t, x)
rms_v = np.sum((term_v - data) ** 2)
rms_jac = 2 * np.sum((term_v - data) * term_v_sens)
print("\n")
print(f"manual: {rms_v} {rms_jac}")
rms_k = np.sum((term_v[k] - data[k]) ** 2)
rms_jac_k = 2 * np.sum((term_v[k] - data[k]) * term_v_sens[k])
print(f"manual [k]: {rms_k} {rms_jac_k}")


vf = jax.vmap(f, in_axes=(0, None))


# Jax:
def square(x):
    return x**2


def rms(t):
    return jnp.sum((vf(t, x) - data) ** 2)


# If the JVP / transpose rules are wrong, then (x*x) and (x**2) will not match
def rms2(t):
    return f(t, x) * f(t, x)


def rms_squared(t):
    return f(t, x) ** 2


def rms_sq_data(t):
    return jnp.sum((f(t, x) - data[k]) ** 2)


print(f"Square(x) {square(3.)} {jax.grad(square)(3.)}")
print(f"RMS: {rms(t_eval)} {jax.grad(rms)(t_eval)}")

print("\nCheck that rms2 and rms_squared match")
print(f"RMS-2: {rms2(t_eval[k])} {jax.grad(rms2)(t_eval[k])}")
print(f"Squared: {rms_squared(t_eval[k])} {jax.grad(rms_squared)(t_eval[k])}")

assert np.allclose(rms_squared(t_eval[k]), rms2(t_eval[k]))
assert np.allclose(jax.grad(rms_squared)(t_eval[k]), jax.grad(rms2)(t_eval[k]))

print("\nCheck that scalar rms (with data) matches manual calculation")
print(f"manual [k]: {rms_k} {rms_jac_k}")
print(f"RMS-sq-data [k]: {rms_sq_data(t_eval[k])} {jax.grad(rms_sq_data)(t_eval[k])}")

assert rms_sq_data(t_eval[k]) == rms_k
assert jax.grad(rms_sq_data)(t_eval[k]) == rms_jac_k

print("\nCheck that vector rms (with data) matches manual calculation")
print(f"manual: {rms_v} {rms_jac}")
print(f"RMS: {rms(t_eval)} {jax.grad(rms)(t_eval)}")

assert rms_v == rms(t_eval)
assert rms_jac == jax.grad(rms)(t_eval)


solver = pybamm.IDAKLUSolver(
    # method, rtol, atol, root_method, root_tol, extrap_tol, output_variables, options
    rtol=1e-6,
    atol=1e-6,
    output_variables=["Terminal voltage [V]"],
)


def jaxsolver(
    # model, t_eval=, inputs=, initial_conditions=, nproc=, calculate_sensitivities=
    model,
    t_eval,  # required arg
    **kwargs,
):
    sim = solver.solve(
        model,
        t_eval,
        **kwargs,
    )
    if kwargs.get("inputs"):
        inputs = kwargs.get("inputs")
        return (
            [sim[out](t_eval) for out in solver.output_variables],
            [sim[out].sensitivities[invar]
                for out in solver.output_variables
                for invar in inputs.keys()]
        )
    else:
        return [sim[var](t_eval) for var in solver.output_variables]


# Solve using IDAKLU directly
if True:
    def fit_fcn(params):
        def sse(fcn, t_eval, inputs, data):
            vec_fcn = jax.vmap(fcn, in_axes=(0, None))
            return jnp.sum((vec_fcn(t_eval, inputs) - data) ** 2, 0)

        inputs = {"Current function [A]": params[0]}
        # return (sse(f, t_eval, inputs, data),
        #         jax.grad(sse(f, t_eval, inputs, data)))

        # term_v, term_v_sens = jaxsolver(
        #     model,
        #     t_eval,
        #     inputs=parameters,
        #     calculate_sensitivities=True,
        # )
        #def sse(t, inputs):
        #    vec_fcn = jax.vmap(f, in_axes=(0, None))
        #    return jnp.sum((vec_fcn(t, inputs) - data) ** 2, 0)
        #    # return jnp.sum((vf(t, d) - data) ** 2)

        f_out, g_out = sse(f, t_eval, inputs, data), jax.grad(sse)(f, t_eval, inputs, data)
        print(f"Params {params=}, RME {f_out=}, Jac {g_out=}")
        return f_out, g_out

    print("Solving without jax")
    bounds = [0.01, 0.6]
    if True:
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            fit_fcn,
            x0,
            bounds=[bounds],
            jac=True,
        )
        print(res)
        print(res.x[0])

# Optimise with scipy (WITHOUT jax)
if False:

    def sse(params, t_eval):
        print("sse ", params)
        term_v, term_v_sens = jaxify_solve(t_eval, params)
        f = np.sum((term_v - data) ** 2)
        g = 2 * np.sum((term_v - data) * term_v_sens)
        return f, g

    bounds = [0.01, 0.6]
    # only primals
    if True:
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            lambda x: sse(x, t_eval)[0],
            x0,
            bounds=[bounds],
        )
        print(res)
        print(res.x[0])
    # with jac/sensitivities
    if True:
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            lambda x: sse(x, t_eval),
            x0,
            jac=True,
            bounds=[bounds],
        )
        print(res)
        print(res.x[0])
        assert np.isclose(res.x[0], inputs["Current function [A]"], atol=1e-3)

# Optimise with scipy (WITH jax)
if False:
    print("\nOptimise with scipy (WITH jax)")

    # only primals
    def sse_jax(params, t_eval):
        print(params)
        d = inputs0.copy()
        d["Current function [A]"] = params
        term_v = vmap_f(t_eval, d)
        f = np.sum((term_v - data) ** 2)
        return f

    bounds = [0.01, 0.6]
    if True:
        print("  only primals")
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            lambda x: sse_jax(x, t_eval),
            x0,
            bounds=[bounds],
        )
        print(res)
        print(f"Result: {res.x[0]}")
        assert np.isclose(res.x[0], inputs["Current function [A]"], atol=1e-2)

    # with jac/sensitivities
    def sse_jax_jac(params, varnames, t_eval):
        def sse(t):
            assert len(params) == len(varnames)
            vf = jax.vmap(f, in_axes=(0, None))
            d = inputs0.copy()
            for ix, name in enumerate(varnames):
                d[name] = params[ix]
            return jnp.sum((vf(t, d) - data) ** 2)

        f_out, g_out = sse(t_eval), jax.grad(sse)(t_eval)
        print(f"Params {params=}, RME {f_out=}, Jac {g_out=}")
        return f_out, g_out

    if True:
        print("  with jac/sensitivities")
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            lambda x: sse_jax_jac(x, ["Current function [A]"], t_eval),
            x0,
            jac=True,
            bounds=[bounds],
        )
        print(res)
        print(f"Result: {res.x[0]}")
        assert np.isclose(res.x[0], inputs["Current function [A]"], atol=1e-3)
