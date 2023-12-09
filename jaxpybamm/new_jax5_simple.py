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
    "Total lithium [mol]",
]

#######################################################################################


if False:

    def f(t, inputs):
        return jnp.array(
            [
                2.0 * inputs[0],
                3.0 * inputs[1],
                inputs[0] * inputs[1],
            ]
        )

    n_in = len(inputs)
    n_out = len(output_variables)

    print("\nf")
    out = f(t_eval[tk], [1.0, 2.0])
    print(type(out))
    print(out)

    print("\nf")
    out = jax.jacfwd(f, argnums=1)(t_eval[tk], [1.0, 2.0])
    print("len of output: ", len(out), type(out))
    for o in out:
        print(o.shape, type(o))
        print(o)

    print("\nf")
    out = jax.jacrev(f, argnums=1)(t_eval[tk], [1.0, 2.0])
    print("len of output: ", len(out), type(out))
    for o in out:
        print(o.shape, type(o))
        print(o)

    print("\njacfwd-jacrev")
    out = jax.jacrev(jax.jacfwd(f, argnums=1), argnums=1)(t_eval[tk], [1.0, 2.0])
    print("len of output: ", len(out), type(out))
    for o in out:
        print(o)


#######################################################################################


if True:

    def f(t, inputs):
        print("f")
        print("  t: ", t)
        print("  inputs: ", inputs)
        out = jnp.array(
            [
                t + 7.0 * inputs["in1"],
                t * t + 3.0 * inputs["in2"],
                t * t * t + inputs["in1"] * inputs["in2"],
            ],
            dtype="float64",
        )
        print("f-out: ", out)
        return out

    varnames = ["out1", "out2", "out3"]

    def isolate_var(f, varname):
        index = varnames.index(varname)

        def f_isolated(*args, **kwargs):
            return f(*args, **kwargs)[index]

        return f_isolated

    n_in = len(inputs)
    n_out = len(output_variables)
    inputs = {"in1": 1.0, "in2": 2.0}
    t_eval = np.arange(20, dtype="float64")
    tk = 5

    print("\nf")
    out = f(t_eval[tk], inputs)
    print(type(out), out)

    print("\nf vmap")
    out = jax.vmap(f, in_axes=(0, None))(t_eval, inputs)
    print(type(out), out)

    print("\njacfwd (wrt t)")
    out = jax.jacfwd(f, argnums=0)(t_eval[tk], inputs)
    print(out)

    print("\njacfwd (wrt t^2)")
    out = jax.jacfwd(jax.jacfwd(f))(t_eval[tk], inputs)
    print(out)

    print("\njacfwd (wrt t^3)")
    out = jax.jacfwd(jax.jacfwd(jax.jacfwd(f)))(t_eval[tk], inputs)
    print(out)

    print("\njvp")
    out = jax.jvp(f, (t_eval[tk], inputs), (1.0, inputs))
    print(out)

    print("\njacfwd (wrt inputs)")
    out = jax.jacfwd(f, argnums=1)(t_eval[tk], inputs)
    print(out)
    print("len of output: ", len(out), type(out))
    for k, o in out.items():
        print(o.shape, type(o))
        print(o)

    print("\njacfwd (wrt inputs) vmap")
    out = jax.vmap(jax.jacfwd(f, argnums=1), in_axes=(0, None))(t_eval, inputs)
    print(out)
    print("len of output: ", len(out), type(out))
    for k, o in out.items():
        print(o.shape, type(o))
        print(o)

    print("\njacrev")
    out = jax.jacrev(f, argnums=1)(t_eval[tk], inputs)
    print("len of output: ", len(out), type(out))
    for k, o in out.items():
        print(o.shape, type(o))
        print(o)

    print("\njacrev vmap")
    out = jax.vmap(jax.jacrev(f, argnums=1), in_axes=(0, None))(t_eval, inputs)
    print("len of output: ", len(out), type(out))
    for k, o in out.items():
        print(o.shape, type(o))
        print(o)

    print("\njacfwd-jacrev")
    out = jax.jacrev(jax.jacfwd(f, argnums=1), argnums=1)(t_eval[tk], inputs)
    print("len of output: ", len(out), type(out))
    for k, o in out.items():
        print(o)

    print("\nisolate_var")
    for varname in varnames:
        out = isolate_var(f, varname)(t_eval[tk], inputs)
        print(varname, type(out), out)

    print("\nisolate_var vmap")
    for varname in varnames:
        out = jax.vmap(isolate_var(f, varname), in_axes=(0, None))(t_eval, inputs)
        print(varname, type(out), out)

    print("\ngrad")
    for varname in varnames:
        out = jax.grad(isolate_var(f, varname), argnums=1)(t_eval[tk], inputs)
        print(varname)
        for k, o in out.items():
            print(o.shape, type(o))
            print(o)

    print("\ngrad vmap")
    for varname in varnames:
        out = jax.vmap(jax.grad(isolate_var(f, varname), argnums=1), in_axes=(0, None))(
            t_eval, inputs
        )
        print(varname)
        for k, o in out.items():
            print(o.shape, type(o))
            print(o)
