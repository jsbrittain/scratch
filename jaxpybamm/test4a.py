import pybamm
import numpy as np
import scipy
from functools import cache
import jax
from jax import lax
from jax.interpreters import ad
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters import batching

model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry
param = model.default_parameter_values
param.update({"Current function [A]": "[input]"})
param.process_geometry(geometry)
param.process_model(model)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
t_eval = np.linspace(0, 3600, 100)
# solver = pybamm.CasadiSolver(mode="fast", rtol=1e-6, atol=1e-6)
solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)

inputs = {'Current function [A]': 0.222}
data = solver.solve(
    model, t_eval,
    inputs=inputs,
)['Terminal voltage [V]'](t_eval)


# IMPLEMENTATION DEFINITION

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def jaxify_solve(t):
    print("jaxify_solve: ", type(t))
    global inputs, t_eval
    if isinstance(inputs, list):
        inputs = inputs[0]
    if not isinstance(t, list) and not isinstance(t, np.ndarray):
        t = [t]
    if isinstance(inputs, np.ndarray):
        inputs = {'Current function [A]': inputs[0]}
    sim = solver.solve(
        model, t_eval,
        inputs=dict(inputs),
        calculate_sensitivities=True,
    )
    term_v = sim['Terminal voltage [V]']
    term_v_sens = sim['Terminal voltage [V]'].sensitivities['Current function [A]']
    if len(t) == 1:
        tk = np.abs(t_eval - t).argmin()
        print("jaxify_solve [exit]: ", term_v(t)[0], term_v_sens[tk])
        return term_v(t)[0], np.array(term_v_sens[tk])[0]
    else:
        print("jaxify_solve [exit]: ", term_v(t_eval)[0], term_v_sens[0])
        return term_v(t_eval), term_v_sens


# JAX PRIMITIVE DEFINITION

f_p = jax.core.Primitive('f')


def f(t):
    return f_p.bind(t)


@f_p.def_impl
def f_impl(t):
    global inputs, t_eval
    """Concrete implementation of Primitive"""
    term_v, term_v_sens = jaxify_solve(t)
    return term_v


@f_p.def_abstract_eval
def f_abstract_eval(t_aval):
    """Abstract evaluation of Primitive
    Takes abstractions of inputs, returned ShapedArray for result of primitive
    """
    print("f_abstract_eval")
    y_aval = jax.core.ShapedArray((1,1), 'float64') #  t_aval.shape, t_aval.dtype)
    return y_aval


def f_batch(args, batch_axes):
    """Batching rule for Primitive
    Takes batched inputs, returns batched outputs and batched axes
    """
    print("f_batch: ", type(args), type(batch_axes))
    t, params, t_eval = args
    # concrete implemenatation provides native batching
    return f(t, [params[0]], t_eval), batch_axes[0]


batching.primitive_batchers[f_p] = f_batch


def f_jvp(primals, tangents):
    """JVP rule for Primitive

    Returns a (primals_out, tangents_out) pair
    primals_out is fun(*primals)
    tangents_out is the Jacobian-vector product of fun @ primals with tangents
    tangents_out has the same Python tree structure and shapes as primals_out
    """
    print("f_jvp: ", type(primals[0]), type(tangents[0]))
    x_t, = primals
    x_dot_t, = tangents
    y, y_dot = jaxify_solve(x_t, inputs, t_eval)
    # y = f(x_t, x_params, x_t_eval)
    # y_dot = f_jvp_p.bind(
    #     (x_t, x_params, x_t_eval),
    #     (x_dot_t, x_dot_params, x_dot_t_eval)
    # )
    return y, y_dot


ad.primitive_jvps[f_p] = f_jvp
f_jvp_p = jax.core.Primitive('f_jvp')


@f_jvp_p.def_impl
def f_jvp_p_eval(primals, tangents):
    print("f_jvp_p_eval: ", type(primals), type(tangents))
    x_t, = primals
    y, y_dot = jaxify_solve(x_t)
    return y_dot[0]


@f_jvp_p.def_abstract_eval
def f_jvp_abstract_eval(primals, tangents):
    print("f_jvp_abstract_eval: ", type(primals), type(tangents))
    x_t_aval, = primals
    x_dot_t_aval, = tangents
    # y_dot_aval = jax.core.ShapedArray(x_dot_t_aval.shape, x_dot_t_aval.dtype)
    y_dot_aval = jax.core.ShapedArray((1, 1), 'float64')
    print("f_jvp_abstract_eval [exit]: ", y_dot_aval)
    return y_dot_aval


def f_jvp_transpose(y_bar):
    # assert ad.is_undefined_primal(x_dot_dummy)
    x_bar = f_vjp_p.bind(x, y_bar)
    print("f_vjp_transpose[exit]: ", x_bar)
    return None, x_bar


ad.primitive_transposes[f_jvp_p] = f_jvp_transpose

# VJP

f_vjp_p = jax.core.Primitive('f_vjp')


@f_vjp_p.def_impl
def f_vjp_impl(x, params):
    t, inputs, t_eval = x
    term_v, term_v_sens = jaxify_solve(t)
    print("f_vjp_impl[exit]: ", term_v_sens[0])
    return term_v_sens[0]


# TEST

t_eval = np.linspace(0.0, 3600, 100)
inputs = {"Current function [A]": 0.222}
x = np.array(list(inputs.values()))

print(f"\nTesting with input: {x=}")

print("\neval with scalar t:")
print(f(t_eval[10]))

# print("\njit with scalar t:")
# print(jax.jit(f)(t_eval[10], x, t_eval))

print("\ngrad with scalar t:")
print(jax.grad(f, argnums=[0])(t_eval[10]))

print("\nvalue_and_grad with scalar t:")
print(jax.value_and_grad(f, argnums=[0])(t_eval[10], x, t_eval))

if True:
    exit(0)

print("\neval with vmap over t:")
print(jax.vmap(f, in_axes=(0, None, None))(t_eval, np.array([x]), t_eval))

print("\ngrad with vmap over t:")
# print(jax.grad(jax.vmap(f, in_axes=(0, None, None)))(t_eval, np.array([x]), t_eval))

print("\nvalue_and_grad with vmap over t:")
#


# Optimise with scipy (WITHOUT jax)
if True:
    def sse(params, t_eval):
        print("sse ", params)
        term_v, term_v_sens = jaxify_solve(t_eval, params, t_eval)
        f = np.sum((term_v - data)**2)
        g = 2 * np.sum((term_v - data) * term_v_sens)
        return f, g

    bounds = [0.01, 0.6]
    # only primals
    if False:
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            lambda x: sse(x, t_eval)[0],
            x0, bounds=[bounds],
        )
        print(res)
        print(res.x[0])
    # with jac/sensitivities
    if False:
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            lambda x: sse(x, t_eval),
            x0, jac=True, bounds=[bounds],
        )
        print(res)
        print(res.x[0])

# Optimise with scipy (WITH jax)
if True:
    print("\nOptimise with scipy (WITH jax)")

    # only primals
    def sse_jax(params, t_eval):
        print(params)
        term_v = vmap_f(t_eval, [params], t_eval)
        f = np.sum((term_v - data)**2)
        return f
    bounds = [0.01, 0.6]
    if False:
        print("  only primals")
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            lambda x: sse_jax(x, t_eval),
            x0, bounds=[bounds],
        )
        print(res)
        print(res.x[0])

    # with jac/sensitivities
    def sse_jax_jac(params, t_eval):
        # TODO: use value_and_grad
        term_v, term_v_sens = value_and_grad_f(t_eval, [params], t_eval)
        f = np.sum((term_v - data)**2)
        g = 2 * np.sum((term_v - data) * term_v_sens)
        return f, g
    if False:
        print("  with jac/sensitivities")
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            lambda x: sse_jax_jac(x, t_eval),
            x0, jac=True, bounds=[bounds],
        )
        print(res)
        print(res.x[0])
