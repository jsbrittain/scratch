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
