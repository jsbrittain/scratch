import pybamm
import numpy as np
import scipy
import jax
import pytest
import logging
from jax.interpreters import ad
from jax.interpreters.mlir import custom_call
from jax._src.lib.mlir.dialects import hlo
from jax.tree_util import tree_flatten, tree_unflatten
from jax.lib import xla_client

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
f_jax = idaklu_solver.jaxify(
    model,
    t_eval,
    output_variables=output_variables,
    inputs=inputs,
    calculate_sensitivities=True,
)

# TEST

f = f_jax
x = inputs

in_axes = (0, None)

print(f"\nTesting with input: {x=}")
# logging.basicConfig(level=logging.INFO)

# Scalar evaluation

k = 5


def test_f_scalar():
    print("\nf (scalar):")
    out = f(t_eval[k], inputs)
    print(out)
    assert np.allclose(
        out,
        np.array([sim[outvar](t_eval[k]) for outvar in output_variables]).transpose(),
    )


def test_f_vector():
    print("\nf (vector):")
    out = f(t_eval, inputs)
    print(out)
    assert np.allclose(
        out, np.array([sim[outvar](t_eval) for outvar in output_variables]).transpose()
    )


def test_f_vmap():
    print("\nf (vmap):")
    out = jax.vmap(f, in_axes=in_axes)(t_eval, x)
    print(out)
    assert np.allclose(
        out, np.array([sim[outvar](t_eval) for outvar in output_variables]).transpose()
    )


def test_getvars_scalar():
    # Get all vars (should mirror above outputs)
    print("\nget_vars (scalar)")
    out = idaklu_solver.get_vars(f, output_variables)(t_eval[k], x)
    print(out)
    assert np.allclose(
        out,
        np.array([sim[outvar](t_eval[k]) for outvar in output_variables]).transpose(),
    )


def test_getvars_vector():
    print("\nget_vars (vector)")
    out = idaklu_solver.get_vars(f, output_variables)(t_eval, x)
    print(out)
    assert np.allclose(
        out, np.array([sim[outvar](t_eval) for outvar in output_variables]).transpose()
    )


def test_getvars_vmap():
    print("\nget_vars (vmap)")
    out = jax.vmap(
        idaklu_solver.get_vars(f, output_variables),
        in_axes=(0, None),
    )(t_eval, x)
    print(out)
    assert np.allclose(
        out, np.array([sim[outvar](t_eval) for outvar in output_variables]).transpose()
    )


def test_getvar_scalar():
    # Per variable checks
    for outvar in output_variables:
        print(f"\nget_var (scalar): {outvar}")
        out = idaklu_solver.get_var(f, outvar)(t_eval[k], x)
        print(out)
        assert np.allclose(out, sim[outvar](t_eval[k]))


def test_getvar_vector():
    for outvar in output_variables:
        print(f"\nget_var (vector): {outvar}")
        out = idaklu_solver.get_var(f, outvar)(t_eval, x)
        print(out)
        assert np.allclose(out, sim[outvar](t_eval))


def test_getvar_vmap():
    for outvar in output_variables:
        print(f"\nget_var (vmap): {outvar}")
        out = jax.vmap(
            idaklu_solver.get_var(f, outvar),
            in_axes=(0, None),
        )(t_eval, x)
        print(out)
        assert np.allclose(out, sim[outvar](t_eval))


# Differentiation rules


def test_jacfwd_scalar():
    print("\njac_fwd (scalar)")
    out = jax.jacfwd(f, argnums=1)(t_eval[k], x)
    print(out)
    flat_out, _ = tree_flatten(out)
    flat_out = np.array([f for f in flat_out]).flatten()
    check = np.array(
        [
            sim[outvar].sensitivities[invar][k]
            for invar in x
            for outvar in output_variables
        ]
    ).transpose()
    assert np.allclose(flat_out, check.flatten())


def test_jacfwd_vmap():
    print("\njac_fwd (vmap)")
    out = jax.vmap(
        jax.jacfwd(f, argnums=1),
        in_axes=(0, None),
    )(t_eval, x)
    print(out)
    flat_out, _ = tree_flatten(out)
    flat_out = np.concatenate(np.array([f for f in flat_out]), 1).transpose().flatten()
    check = np.array(
        [sim[outvar].sensitivities[invar] for invar in x for outvar in output_variables]
    )
    assert np.allclose(flat_out, check.flatten())


def test_jacrev_scalar():
    print("\njac_rev (scalar)")
    out = jax.jacrev(f, argnums=1)(t_eval[k], x)
    print(out)
    flat_out, _ = tree_flatten(out)
    flat_out = np.array([f for f in flat_out]).flatten()
    check = np.array(
        [
            sim[outvar].sensitivities[invar][k]
            for invar in x
            for outvar in output_variables
        ]
    ).transpose()
    assert np.allclose(flat_out, check.flatten())


def test_jacrev_vmap():
    print("\njac_rev (vmap)")
    out = jax.vmap(
        jax.jacrev(f, argnums=1),
        in_axes=(0, None),
    )(t_eval, x)
    print(out)
    flat_out, _ = tree_flatten(out)
    flat_out = np.concatenate(np.array([f for f in flat_out]), 1).transpose().flatten()
    check = np.array(
        [sim[outvar].sensitivities[invar] for invar in x for outvar in output_variables]
    )
    assert np.allclose(flat_out, check.flatten())


# Get all vars (should mirror above outputs)


def test_jacfwd_scalar_getvars():
    print("\njac_fwd (scalar) get_vars")
    out = jax.jacfwd(idaklu_solver.get_vars(f, output_variables), argnums=1)(
        t_eval[k], x
    )
    print(out)
    flat_out, _ = tree_flatten(out)
    flat_out = np.array([f for f in flat_out]).flatten()
    check = np.array(
        [
            sim[outvar].sensitivities[invar][k]
            for invar in x
            for outvar in output_variables
        ]
    ).transpose()
    assert np.allclose(flat_out, check.flatten())


def test_jacfwd_scalar_getvar():
    for outvar in output_variables:
        print(f"\njac_fwd (scalar) get_var: {outvar}")
        out = jax.jacfwd(idaklu_solver.get_var(f, outvar), argnums=1)(
            t_eval[k], x
        )
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar][k]
                for invar in x
            ]
        ).transpose()
        assert np.allclose(flat_out, check.flatten()), f"Got: {flat_out}\nExpected: {check}"


def test_jacfwd_vmap_getvars():
    print("\njac_fwd (vmap) getvars")
    out = jax.vmap(
        jax.jacfwd(idaklu_solver.get_vars(f, output_variables), argnums=1),
        in_axes=(0, None),
    )(t_eval, x)
    print(out)
    flat_out, _ = tree_flatten(out)
    flat_out = np.concatenate(np.array([f for f in flat_out]), 1).transpose().flatten()
    check = np.array(
        [sim[outvar].sensitivities[invar] for invar in x for outvar in output_variables]
    )
    assert np.allclose(flat_out, check.flatten())


def test_jacfwd_vmap_getvar():
    for outvar in output_variables:
        print(f"\njac_fwd (vmap) getvar: {outvar}")
        out = jax.vmap(
            jax.jacfwd(idaklu_solver.get_var(f, outvar), argnums=1),
            in_axes=(0, None),
        )(t_eval, x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 0).transpose().flatten()
        check = np.array(
            [sim[outvar].sensitivities[invar] for invar in x]
        )
        assert np.allclose(flat_out, check.flatten()), f"Got: {flat_out}\nExpected: {check}"


def test_jacrev_scalar_getvars():
    print("\njac_rev (scalar) getvars")
    out = jax.jacrev(idaklu_solver.get_vars(f, output_variables), argnums=1)(
        t_eval[k], x
    )
    print(out)
    flat_out, _ = tree_flatten(out)
    flat_out = np.array([f for f in flat_out]).flatten()
    check = np.array(
        [
            sim[outvar].sensitivities[invar][k]
            for invar in x
            for outvar in output_variables
        ]
    ).transpose()
    assert np.allclose(flat_out, check.flatten())


def test_jacrev_scalar_getvar():
    for outvar in output_variables:
        print(f"\njac_rev (scalar) getvar: {outvar}")
        #out = jax.jacrev(idaklu_solver.get_var(f, outvar), argnums=1)(t_eval[k], x)
        out = jax.jacrev(idaklu_solver.get_vars(f, [output_variables[0]]), argnums=1)(t_eval[k], x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(
            [
                sim[outvar].sensitivities[invar][k]
                for invar in x
            ]
        ).transpose()
        assert np.allclose(flat_out, check.flatten())


def test_jacrev_vmap_getvars():
    print("\njac_rev (vmap) getvars")
    out = jax.vmap(
        jax.jacrev(idaklu_solver.get_vars(f, output_variables), argnums=1),
        in_axes=(0, None),
    )(t_eval, x)
    print(out)
    flat_out, _ = tree_flatten(out)
    flat_out = np.concatenate(np.array([f for f in flat_out]), 1).transpose().flatten()
    check = np.array(
        [sim[outvar].sensitivities[invar] for invar in x for outvar in output_variables]
    )
    assert np.allclose(flat_out, check.flatten())


def test_jacrev_vmap_getvar():
    for outvar in output_variables:
        print(f"\njac_rev (vmap) getvar: {outvar}")
        out = jax.vmap(
            jax.jacrev(idaklu_solver.get_var(f, outvar), argnums=1),
            in_axes=(0, None),
        )(t_eval, x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).transpose().flatten()
        check = np.array(
            [sim[outvar].sensitivities[invar] for invar in x]
        )
        assert np.allclose(flat_out, check.flatten())


# Per variable checks


def test_grad_scalar_getvar():
    for outvar in output_variables:
        print(f"\ngrad (scalar) getvar: {outvar}")
        out = jax.grad(
            idaklu_solver.get_var(f, outvar),
            argnums=1,
        )(
            t_eval[k], x
        )  # output should be a dictionary of inputs
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).transpose().flatten()
        check = np.array(
            [sim[outvar].sensitivities[invar][k] for invar in x]
        )
        assert np.allclose(flat_out, check.flatten())


def test_grad_vmap_getvar():
    for outvar in output_variables:
        print(f"\ngrad (vmap) getvars: {outvar}")
        out = jax.vmap(
            jax.grad(
                idaklu_solver.get_var(f, outvar),
                argnums=1,
            ),
            in_axes=(0, None),
        )(t_eval, x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.concatenate(np.array([f for f in flat_out]), 1).transpose().flatten()
        check = np.array(
            [sim[outvar].sensitivities[invar] for invar in x]
        )
        assert np.allclose(flat_out, check.flatten())


if __name__ == "__main__":
    testlist = [
        test_f_scalar,
        test_f_vector,
        test_f_vmap,
        test_getvars_scalar,
        test_getvars_vector,
        test_getvars_vmap,
        test_getvar_scalar,
        test_getvar_vector,
        test_getvar_vmap,
        test_jacfwd_scalar,
        test_jacfwd_vmap,
        test_jacrev_scalar,
        test_jacrev_vmap,
        test_jacfwd_scalar_getvars,
        test_jacfwd_scalar_getvar,
        test_jacfwd_vmap_getvars,
        test_jacfwd_vmap_getvar,
        test_jacrev_scalar_getvars,
        test_jacrev_scalar_getvar,
        test_jacrev_vmap_getvars,
        test_jacrev_vmap_getvar,
        test_grad_scalar_getvar,
        test_grad_vmap_getvar,
    ]
    testlist = [
        test_jacrev_scalar_getvar,
        test_jacrev_vmap_getvar,
        test_grad_scalar_getvar,
        test_grad_vmap_getvar,
    ]

    for test in testlist:
        print(f"\nRunning test: {test.__name__}")
        test()
