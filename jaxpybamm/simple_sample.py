import pybamm
import numpy as np
import scipy


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

data = solver.solve(
    model, t_eval,
    inputs={'Current function [A]': 0.222},
)['Terminal voltage [V]'](t_eval)


def sse_jac(params):
    print("params", params)
    sim = solver.solve(
        model, t_eval,
        inputs={'Current function [A]': params[0]},
        calculate_sensitivities=True,
    )
    term_v = sim['Terminal voltage [V]'](t_eval)
    term_v_sens = sim['Terminal voltage [V]'].sensitivities['Current function [A]']

    f = np.sum((term_v - data)**2)
    g = 2 * np.sum((term_v - data) * term_v_sens)
    return f, g


bounds = [0.01, 0.6]
x0 = np.random.uniform(*bounds)
res = scipy.optimize.minimize(
    sse_jac, x0, jac=True, bounds=[bounds],
)
print(res)
print(res.x[0])
