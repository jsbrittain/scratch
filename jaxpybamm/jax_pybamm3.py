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
if cpu_ops_spec:
    cpu_ops = importlib.util.module_from_spec(cpu_ops_spec)
    if cpu_ops_spec.loader:
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
f_jax = idaklu_solver.jaxify(
    model,
    t_eval,
    output_variables=output_variables,
    inputs=inputs,
    calculate_sensitivities=True,
)

print(" ")
print(" ")

print(f_jax(t_eval, inputs))
f = f_jax

print(" ")
print(" ")

# TEST

# t_eval = np.linspace(0.0, 360, 10)
x = inputs  # np.array(list(inputs.values()))

d_axes = None  # dict(zip(inputs.keys(), repeat(None)))  # can also be dict
in_axes = (0, d_axes)
check_asserts = True

print(f"\nTesting with input: {x=}")

# Scalar evaluation

print("\neval with non-scalar t:")  # calls f
print(f(t_eval, inputs))

k = 5
print("\neval with scalar t:")  # calls f
print(f(t_eval[k], inputs))

print("\neval with vmap over t:")  # calls f
vmap_f = jax.vmap(f, in_axes=in_axes)
print(vmap_f(t_eval, x))

# print("\njacfwd scalar:")  # calls f
# print(jax.jacfwd(f)(t_eval[k], x))

# print("\njacfwd vector:")  # calls f
# print(jax.jacfwd(f)(t_eval, x))

# print("\njacrev scalar:")  # calls f
# print(jax.jacrev(f)(t_eval[k], x))

# print("\njacrev vector:")  # calls f
# print(jax.jacrev(f)(t_eval, x))

# print("\ngrad with scalar t:")  # calls f
# g = jax.grad(f, argnums=0)(t_eval[k], x)
# print(g)

print('get_var (scalar)')
for outvar in output_variables:
    out = idaklu_solver.get_var(f, outvar)(t_eval[k], x)
    print(f' {outvar=} {out=}')
    if check_asserts:
        check = np.array(sim[outvar](t_eval[k]))
        assert np.allclose(out, check)
        # This is an alternative way to produce exactly the same result
        #assert np.allclose(out, f(t_eval[k], x)[output_variables.index(outvar)])

print('get_vars (scalar)')
out = idaklu_solver.get_vars(f, output_variables)(t_eval[k], x)
print(f' {out=}')
check = np.array([sim[outvar](t_eval[k]) for outvar in output_variables])
print(f' {check=}')
if check_asserts:
    assert np.allclose(out, check)

print('get_var (vector)')
for outvar in output_variables:
    out = idaklu_solver.get_var(f, outvar)(t_eval, x)
    print(f' {outvar=} {out=}')
    if check_asserts:
        assert np.allclose(out, sim[outvar](t_eval))

print('get_vars (vector)')
out = idaklu_solver.get_vars(f, output_variables)(t_eval, x)
print(f' {out=}')
check = np.array([sim[outvar](t_eval) for outvar in output_variables]).transpose()
if check_asserts:
    assert np.allclose(out, check)

print('get_var (vmap)')
for outvar in output_variables:
    out = jax.vmap(
        idaklu_solver.get_var(f, outvar),
        in_axes=(0, None),
    )(t_eval, x)
    print(f' {outvar=} {out=}')
    if check_asserts:
        assert np.allclose(out, sim[outvar](t_eval))

print('get_vars (vmap)')
out = idaklu_solver.get_vars(
    jax.vmap(f, in_axes=in_axes),
    output_variables
)(t_eval, x)
print(f' {out=}')
check = np.array([sim[outvar](t_eval) for outvar in output_variables]).transpose()
if check_asserts:
    assert np.allclose(out, check)

#print('grad get_var (scalar)')
#for outvar in output_variables:
#    for invar in inputs:
#        out = jax.grad(
#            idaklu_solver.get_var(f, outvar)
#        )(t_eval[k], x)
#        print(f' {outvar=} {invar=} {out=}')
#        if check_asserts:
#            assert np.allclose(out, sim[outvar].sensitivities[invar][k])

print('grad get_var (vmap)')
for outvar in output_variables:
    for invar in inputs:
        out = jax.vmap(
            jax.grad(idaklu_solver.get_var(f, outvar)),
            in_axes=(0, None),
        )(t_eval, x)
        print(f' {outvar=} {invar=} {out=}')
        check = np.array(sim[outvar].sensitivities[invar]).transpose()
        print(check)
        if check_asserts:
            assert np.allclose(out, check)

#print('value_and_grad get_var (scalar)')
#for outvar in output_variables:
#    for invar in inputs:
#        out_p, out_t = jax.value_and_grad(
#            idaklu_solver.get_var(f, outvar)
#        )([t_eval[k]], x)
#        print(f' {outvar=} {invar=} {out_p=} {out_t=}')
#        if check_asserts:
#            assert np.allclose(out_p, sim[outvar](t_eval[k]))
#            assert np.allclose(out_t, sim[outvar].sensitivities[invar][k])

print('grad get_value_and_grad (vmap)')
for outvar in output_variables:
    for invar in inputs:
        out_v, out_g = jax.vmap(
            jax.value_and_grad(idaklu_solver.get_var(f, outvar)),
            in_axes=(0, None),
        )(t_eval, x)
        print(f' {outvar=} {invar=} {out_v=} {out_g=}')
        check_v = np.array(sim[outvar](t_eval))
        check_p = np.array(sim[outvar].sensitivities[invar]).transpose()
        if check_asserts:
            assert np.allclose(out_v, check_v)
            assert np.allclose(out_g, check_p)

exit(0)

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


# Solve using IDAKLU directly
if True:
    def fit_fcn(params):
        def sse(t_eval, inputs, data):
            return jnp.sum((vec_fcn(t_eval, inputs) - data) ** 2, 0)

        inputs = {"Current function [A]": params[0]}
        f_out, g_out = sse(t_eval, inputs, data), jax.grad(sse)(t_eval, inputs, data)
        print(f"Params {params=}, RME {f_out=}, Jac {g_out=}")
        return f_out, g_out

    print("Solving without jax")
    bounds = [0.01, 0.6]
    if True:
        vec_fcn = jax.vmap(f, in_axes=(0, None))
        x0 = np.random.uniform(*bounds)
        res = scipy.optimize.minimize(
            fit_fcn,
            x0,
            bounds=[bounds],
            jac=True,
        )
        print(res)
        print(res.x[0])
