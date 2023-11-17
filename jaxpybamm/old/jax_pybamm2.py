import pybamm
import numpy as np
import scipy
from functools import cache
import jax
import logging
from jax import lax
from jax import numpy as jnp
from jax.interpreters import ad
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters import batching
from jax._src.lib.mlir.dialects import hlo


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
# logger.setLevel(logger.DEBUG)

inputs = {
    "Current function [A]": 0.222,
    #    'Separator porosity': 0.3,
}

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


# IMPLEMENTATION DEFINITION
#
# See https://github.com/google/jax/blob/main/jax/_src/core.py
# for Primitive class definition


def jaxify_solve(t, *inputs):
    logger.info("jaxify_solve: ", type(t), type(inputs))
    if not isinstance(t, list) and not isinstance(t, np.ndarray):
        t = [t]
    if isinstance(inputs, list):
        inputs = inputs[0]
    if (
        isinstance(inputs, np.ndarray)
        or isinstance(inputs, list)
        or isinstance(inputs, tuple)
    ):
        inputs = {
            "Current function [A]": inputs[0],
            #        'Separator porosity': inputs[1],
        }
    sim = solver.solve(
        model,
        t_eval,
        inputs=dict(inputs),
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
    if len(inputs) == 1 and isinstance(inputs[0], dict):
        # dictionary of inputs supplied by user
        # inputs_vector = np.array(list(inputs[0].values()))
        bind_point = f_p.bind(t, *inputs[0].values())
    else:
        # other primitive functions (e.g. jvp) call with a vector of inputs
        bind_point = f_p.bind(t, *inputs)
    logger.debug("f [exit]: ", (type(bind_point), bind_point))
    return bind_point


@f_p.def_impl
def f_impl(*args):
    """Concrete implementation of Primitive"""
    logger.info("f_impl: ")
    term_v, term_v_sens = jaxify_solve(*args)
    logger.debug("f_impl [exit]: ", (type(term_v), term_v))
    return term_v


# @f_p.def_abstract_eval
# def f_abstract_eval(t_aval, params_aval, t_eval_aval):
#     """Abstract evaluation of Primitive
#     Takes abstractions of inputs, returned ShapedArray for result of primitive
#     """
#     logger.info("f_abstract_eval")
#     y_aval = jax.core.ShapedArray(t_aval.shape, t_aval.dtype)
#     return y_aval


def f_batch(args, batch_axes):
    """Batching rule for Primitive
    Takes batched inputs, returns batched outputs and batched axes
    """
    logger.info("f_batch: ", type(args), type(batch_axes))
    # concrete implemenatation provides native batching
    # return f(t, np.array([params[0]]), t_eval), batch_axes[0]
    return f(*args), batch_axes[0]


batching.primitive_batchers[f_p] = f_batch


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
    y = f(*primals)  # primals_out = fun(*primals); (unnecessary call?)
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
    t_dot = args[len(args) // 2]  # first tangent
    y_dot_aval = jax.core.ShapedArray(t_dot.shape, t_dot.dtype)
    logger.debug("f_jvp_abstract_eval [exit]: ", (type(y_dot_aval), y_dot_aval))
    return y_dot_aval


def f_jvp_transpose(y_bar, *args):
    """JVP transpose rule

    Args:
        y_bar: cotangent of the output of the primitive

    """
    logger.info("f_jvp_transpose: ", type(y_bar), len(args))
    # assert ad.is_undefined_primal(x_dot_dummy)
    primals = args[: len(args) // 2]
    x_bar = f_vjp_p.bind(y_bar, *primals)
    logger.debug("j_jvp_transpose [exit]: ", (type(x_bar), x_bar), (type(y_bar), y_bar))
    # print("j_jvp_transpose [exit]: ", (type(x_bar), x_bar), (type(y_bar), y_bar))
    primals_out = (None,) * len(primals)
    tangents_out = jnp.dot(y_bar, x_bar), *(
        (None,) * (len(primals) - 1)
    )  # TODO: Generalise function
    return *primals_out, *tangents_out


ad.primitive_transposes[f_jvp_p] = f_jvp_transpose

# VJP / Reverse-mode autodiff / v^T.J len(v)=num_outputs / len(return)=num_inputs

f_vjp_p = jax.core.Primitive("f_vjp")


@f_vjp_p.def_impl
def f_vjp_impl(y_bar, *args):
    """
    Takes y_bar and primals
    """
    logger.info("f_vjp_impl: ")
    term_v, term_v_sens = jaxify_solve(*args)
    logger.debug("f_vjp_impl [exit]: ", (type(term_v_sens), term_v_sens))
    return term_v_sens


def f_vjp_batch(args, batch_axes):
    logger.info("f_vjp_batch: ", type(args), type(batch_axes))
    # y_bar, t, params = args
    # concrete implemenatation provides native batching
    y_bar = args[0]
    primals = args[1:]
    term_v, term_v_sens = jaxify_solve(*primals)
    term_v_sens = np.array(term_v_sens)
    logger.debug("f_vjp_batch [exit]: ", (type(term_v_sens), term_v_sens))
    print("y_bar: ", y_bar)
    return term_v_sens, batch_axes[0]


batching.primitive_batchers[f_vjp_p] = f_vjp_batch


# TEST

# t_eval = np.linspace(0.0, 360, 10)
x = inputs  #  np.array(list(inputs.values()))

# Scalar evaluation

k = 5
print("\neval with scalar t:")
print(f(t_eval[k], inputs))

print("\ngrad with scalar t:")
print(jax.grad(f)(t_eval[k], x))

print("\nvalue_and_grad with scalar t:")
value_and_grad_f = jax.value_and_grad(f)
print(value_and_grad_f(t_eval[k], x))

# Vector evaluation

print("\neval with vmap over t:")
vmap_f = jax.vmap(f, in_axes=(0, None))
print(vmap_f(t_eval, x))

print("\ngrad with vmap over t:")
vmap_grad = jax.vmap(jax.grad(f), in_axes=(0, None))
print(vmap_grad(t_eval, x))

exit(0)

print("\nvalue_and_grad with vmap over t:")
vmap_vg = jax.vmap(jax.value_and_grad(f), in_axes=(0, None))
print(vmap_vg(t_eval, x))

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


exit(0)


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


# Optimise with scipy (WITHOUT jax)
if False:

    def sse(params, t_eval):
        print("sse ", params)
        term_v, term_v_sens = jaxify_solve(t_eval, params, t_eval)
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
if True:
    print("\nOptimise with scipy (WITH jax)")

    # only primals
    def sse_jax(params, t_eval):
        print(params)
        term_v = vmap_f(t_eval, [params], t_eval)
        f = np.sum((term_v - data) ** 2)
        return f

    bounds = [0.01, 0.6]
    if False:
        print("  only primals")
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            lambda x: sse_jax(x, t_eval),
            x0,
            bounds=[bounds],
        )
        print(res)
        print(f"Result: {res.x[0]}")
        assert np.isclose(res.x[0], inputs["Current function [A]"], atol=1e-3)

    # with jac/sensitivities
    def sse_jax_jac(params, t_eval):
        def sse(t):
            vf = jax.vmap(f, in_axes=(0, None, None))
            return jnp.sum((vf(t, np.append(params, 3.0), t_eval) - data) ** 2)

        f_out, g_out = sse(t_eval), jax.grad(sse)(t_eval)
        print(f"Params {params=}, RME {f_out=}, Jac {g_out=}")
        return f_out, g_out

    if True:
        print("  with jac/sensitivities")
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            lambda x: sse_jax_jac(x, t_eval),
            x0,
            jac=True,
            bounds=[bounds],
        )
        print(res)
        print(f"Result: {res.x[0]}")
        assert np.isclose(res.x[0], inputs["Current function [A]"], atol=1e-3)
