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
import jax.numpy as jnp

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

# Get jax expression for IDAKLU solver
output_variables = [
    "Terminal voltage [V]",
    "Discharge capacity [A.h]",
    "Loss of lithium inventory [%]",
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


def test_jacfwd_vector():
    print("\njac_fwd (vector)")
    out = jax.jacfwd(f, argnums=1)(t_eval, x)
    print(out)
    flat_out, _ = tree_flatten(out)
    flat_out = np.concatenate(np.array([f for f in flat_out]), 1).transpose().flatten()
    check = np.array(
        [sim[outvar].sensitivities[invar] for invar in x for outvar in output_variables]
    )
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


def test_jacrev_vector():
    print("\njac_rev (vector)")
    out = jax.jacrev(f, argnums=1)(t_eval, x)
    print(out)
    flat_out, _ = tree_flatten(out)
    flat_out = np.concatenate(np.array([f for f in flat_out]), 1).transpose().flatten()
    check = np.array(
        [sim[outvar].sensitivities[invar] for invar in x for outvar in output_variables]
    )
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
        out = jax.jacfwd(idaklu_solver.get_var(f, outvar), argnums=1)(t_eval[k], x)
        print(out)
        flat_out, _ = tree_flatten(out)
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(
            [sim[outvar].sensitivities[invar][k] for invar in x]
        ).transpose()
        assert np.allclose(
            flat_out, check.flatten()
        ), f"Got: {flat_out}\nExpected: {check}"


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
        flat_out = (
            np.concatenate(np.array([f for f in flat_out]), 0).transpose().flatten()
        )
        check = np.array([sim[outvar].sensitivities[invar] for invar in x])
        assert np.allclose(
            flat_out, check.flatten()
        ), f"Got: {flat_out}\nExpected: {check}"


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
        out = jax.jacrev(idaklu_solver.get_var(f, outvar), argnums=1)(t_eval[k], x)
        flat_out, _ = tree_flatten(out)
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(
            [sim[outvar].sensitivities[invar][k] for invar in x]
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
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array([sim[outvar].sensitivities[invar] for invar in x])
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
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array([sim[outvar].sensitivities[invar][k] for invar in x])
        print("expected: ", check.flatten())
        print("got: ", flat_out)
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
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array([sim[outvar].sensitivities[invar] for invar in x])
        assert np.allclose(flat_out, check.flatten())


def test_value_and_grad_scalar():
    for outvar in output_variables:
        print(f"\nvalue_and_grad (scalar): {outvar}")
        primals, tangents = jax.value_and_grad(
            idaklu_solver.get_var(f, outvar),
            argnums=1,
        )(t_eval[k], x)
        print(primals)
        flat_p, _ = tree_flatten(primals)
        flat_p = np.array([f for f in flat_p]).flatten()
        check = np.array(sim[outvar].data[k])
        assert np.allclose(flat_p, check.flatten())
        print(tangents)
        flat_t, _ = tree_flatten(tangents)
        flat_t = np.array([f for f in flat_t]).flatten()
        check = np.array([sim[outvar].sensitivities[invar][k] for invar in x])
        assert np.allclose(flat_t, check.flatten())


def test_value_and_grad_vmap():
    for outvar in output_variables:
        print(f"\nvalue_and_grad (vmap): {outvar}")
        primals, tangents = jax.vmap(
            jax.value_and_grad(
                idaklu_solver.get_var(f, outvar),
                argnums=1,
            ),
            in_axes=(0, None),
        )(t_eval, x)
        print(primals)
        flat_p, _ = tree_flatten(primals)
        flat_p = np.array([f for f in flat_p]).flatten()
        check = np.array(sim[outvar].data)
        assert np.allclose(flat_p, check.flatten())
        print(tangents)
        flat_t, _ = tree_flatten(tangents)
        flat_t = np.array([f for f in flat_t]).flatten()
        check = np.array([sim[outvar].sensitivities[invar] for invar in x])
        assert np.allclose(flat_t, check.flatten())


def test_jax_vars():
    print("\njax_vars")
    out = idaklu_solver.jax_value()
    print(out)
    for outvar in output_variables:
        flat_out, _ = tree_flatten(out[outvar])
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array(sim[outvar].data)
        assert np.allclose(flat_out, check.flatten()), \
            f"{outvar}: Got: {flat_out}\nExpected: {check}"


def test_jax_grad():
    print("\njax_grad")
    out = idaklu_solver.jax_grad()
    print(out)
    for outvar in output_variables:
        flat_out, _ = tree_flatten(out[outvar])
        flat_out = np.array([f for f in flat_out]).flatten()
        check = np.array([sim[outvar].sensitivities[invar] for invar in x])
        assert np.allclose(flat_out, check.flatten()), \
            f"{outvar}: Got: {flat_out}\nExpected: {check}"


def test_grad_wrapper_sse():
    print("\ngrad_wrapper")
    vf = jax.vmap(
        idaklu_solver.get_var(f, "Terminal voltage [V]"),
        in_axes=(0, None)
    )

    # Use surrogate for experimental data
    data = sim["Terminal voltage [V]"](t_eval)

    # Define SSE function to minimise - note that although f returns a vector over time,
    # sse() returns a scalar so can be passed to grad().
    def sse(t, inputs):
        return jnp.sum((vf(t_eval, inputs) - data) ** 2)

    # Create an imperfect prediction
    inputs_pred = inputs.copy()
    inputs_pred["Current function [A]"] = 0.150
    sim_pred = idaklu_solver.solve(
        model,
        t_eval,
        inputs=inputs_pred,
        calculate_sensitivities=True,
    )
    pred = sim_pred["Terminal voltage [V]"]

    # Check value against actual SSE
    sse_actual = np.sum((pred(t_eval) - data) ** 2)
    print(f"SSE: {sse(t_eval, inputs_pred)}")
    print(f"SSE-actual: {sse_actual}")
    flat_out, _ = tree_flatten(sse(t_eval, inputs_pred))
    flat_out = np.array([f for f in flat_out]).flatten()
    flat_check_val, _ = tree_flatten(sse_actual)
    assert np.allclose(flat_out, flat_check_val, 1e-3), \
        f"Got: {flat_out}\nExpected: {flat_check_val}"

    # Check grad against actual
    sse_grad_actual = {}
    for k, v in inputs_pred.items():
        sse_grad_actual[k] = 2 * np.sum((pred(t_eval) - data) * pred.sensitivities[k])
    sse_grad = jax.grad(sse, argnums=1)(t_eval, inputs_pred)
    print(f"SSE-grad: {sse_grad}")
    print(f"SSE-grad-actual: {sse_grad_actual}")
    flat_out, _ = tree_flatten(sse_grad)
    flat_out = np.array([f for f in flat_out]).flatten()
    flat_check_grad, _ = tree_flatten(sse_grad_actual)
    assert np.allclose(flat_out, flat_check_grad, 1e-3), \
        f"Got: {flat_out}\nExpected: {flat_check_grad}"

    # Check value_and_grad return
    sse_val, sse_grad = jax.value_and_grad(sse, argnums=1)(t_eval, inputs_pred)
    flat_sse_grad, _ = tree_flatten(sse_grad)
    flat_sse_grad = np.array([f for f in flat_sse_grad]).flatten()
    assert np.allclose(sse_val, flat_check_val, 1e3), \
        f"Got: {sse_val}\nExpected: {flat_check_val}"
    assert np.allclose(flat_sse_grad, flat_check_grad, 1e3), \
        f"Got: {sse_grad}\nExpected: {sse_grad}"


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
        test_jacfwd_vector,
        test_jacfwd_vmap,
        test_jacrev_scalar,
        # test_jacrev_vector,
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
        test_value_and_grad_scalar,
        test_value_and_grad_vmap,
        test_jax_vars,
        test_jax_grad,
        test_grad_wrapper_sse,
    ]
    if 0:
        testlist = [
            test_jax_vars,
        ]

    for test in testlist:
        print(f"\nRunning test: {test.__name__}")
        test()
