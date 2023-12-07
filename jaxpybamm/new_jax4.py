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
#    "Discharge capacity [A.h]",
]

#######################################################################################


calculate_sensitivities = True


if False:
    def isolate_var(fcn, varname):
        print('  isolate_var')
        index = output_variables.index(varname)

        def impl(t, *args, **kwargs):
            print('    isolate_var_impl')
            out = jnp.array(fcn(t, *args, **kwargs))
            # Vectors always revert to (N,), meaning we don't know if they are multiple
            # time points for a single variable or multiple variables at a single time point
            if t.ndim == 0:
                t = jnp.array([t])
            if out.ndim == 1 and len(t) == 1:
                # one or more variables at a single time point
                out = out.reshape((1, out.shape[0]))
                ...
            elif out.ndim == 1 and len(t) > 1:
                # one variable at multiple time points
                out = out.reshape((len(t), 1))
            else:
                # multiple variables at multiple time points
                ...
            if len(output_variables) > 1:
                out = out[:, index]
            if out.shape == (1,):
                out = out[0]  # scalar
            return out
        return impl


f_p = jax.core.Primitive("f")
# f_p.multiple_results = True  # return a vector (of time samples)

from jax.tree_util import tree_flatten, tree_unflatten


@jax.custom_jvp
def f(t, inputs, **kwargs):
    values = jnp.array(list(inputs.values()))
    out = f_p.bind(t, values, **kwargs)
    #inputs, pytree_def = tree_flatten(inputs)
    #out = f_p.bind(t, inputs, **kwargs)
    return out



class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


@lru_cache(maxsize=1)  # cache the current solve for reuse
def cached_solve(model, t_eval, *args, **kwargs):
    # reconstruct time vector
    t_eval = jnp.array(list(t_eval.values()))
    return solver.solve(model, t_eval, *args, **kwargs)


@f_p.def_impl
def f_impl(t, inputs, *, invar=None):
    print('  f_impl')
    print('    t: ', t)
    print('    inputs: ', inputs)
    # convert scalar time values to vector
    if t.ndim == 0:
        t = jnp.array([t], dtype=jnp.float64)
    # reconstruct input dictionary (also make hashable for caching the solve)
    d = hashabledict()
    for ix, key in enumerate(inputs0.keys()):
        d[key] = inputs[ix].item()
    # make time vector hashable (for caching the solve)
    t_dict = hashabledict()
    for ix, tval in enumerate(t_eval):
        t_dict[ix] = tval.item()
    # Debug
    if False:
        print("Solver (f_impl):")
        print("  t: ", t, type(t))
        print("  inputs: ", d)
    # Solve (or retrieve from cache)
    sim = cached_solve(
        model,
        t_dict,
        inputs=d,
        calculate_sensitivities=True,
    )
    # Primals
    out = jnp.array([jnp.array(sim[var](t)) for var in output_variables]).transpose()
    if invar is not None:
        # Return sensitivities
        tk = [k for k, tval in enumerate(t_eval) if tval >= t[0] and tval <= t[-1]]
        out_sens = jnp.reshape(
            jnp.array([
                jnp.array(sim[outvar].sensitivities[invar][tk])
                for outvar in output_variables
            ]).transpose(),
            out.shape)
        return out, out_sens

    return out


if True:
    @f_p.def_abstract_eval
    def f_abstract_eval(t, *args):
        """Abstract evaluation of Primitive
        Takes abstractions of inputs, returned ShapedArray for result of primitive
        """
        assert False, "f_abstract_eval"
        print('  f_abstract_eval')
        y_aval = jax.core.ShapedArray(t.shape, t.dtype)
        return y_aval


if True:
    def f_batch(args, batch_axes, **kwargs):
        print('  f_batch')
        tvec = args[0]
        if tvec.ndim == 0:
            tvec = jnp.array([tvec])
        out = jnp.array(
            list(map(lambda t: f_p.bind(t, *args[1:], **kwargs), tvec))
        ).squeeze()
        return out, batch_axes[0]
    
    batching.primitive_batchers[f_p] = f_batch


@f.defjvp
def f_jvp(primals, tangents):
    print('  f_jvp')
    print('   primals: ', primals)
    print('   tangents: ', tangents)
    # assert type(tangents[0]) is ad.Zero, "We do not currently support time derivatives"
    assert type(tangents[1]) is not ad.Zero, "We currently only support input variable derivatives"

    # List of input variable names
    invars = list(inputs0.keys())

    # Primals
    values = jnp.array(list(inputs.values()))
    p = (primals[0], values, *primals[2:])
    print(p)
    y = f_p.bind(*p, invar=None)

    # Tangents
    tangent = tangents[1]  # tangent = matrix wrt each output, e.g. [[1, 0], [0, 1]]
    if isinstance(tangent, dict):
        tangent = jnp.array(list(tangent.values()))
    if hasattr(tangent, 'val'):
        tangent = tangent.val
    print(' ')
    print('tangent: ', tangent)
    if tangent.ndim == 1:
        tangent = jnp.array([tangent])
    y_dot = jnp.zeros((1, tangent.shape[0]))
    print('y_dot shape: ', y_dot.shape)
    for row_index, row in enumerate(tangent):
        print('row_index: ', row_index)
        print('row: ', row)
        # Sum over all input variables
        invar = invars[row_index]
        # solver returns sensitivities for all outputs given an input
        _, sens = f_p.bind(*p, invar=invar)
        if hasattr(row, 'val'):
            row_val = row.val
        else:
            row_val = row
        print('row_val: ', row_val)
        print('sens: ', sens[0])
        print('np.dot(row_val, sens[0]): ', np.dot(row_val, sens[0]))
        y_dot = y_dot.at[:, row_index].set(np.dot(row_val, sens[0]))
    print('y: ', y)
    print('y_dot: ', y_dot)
    return y, y_dot,


# ad.primitive_jvps[f_p] = f_jvp


def f_jvp_transpose(y_bar, *args):
    assert False, "f_jvp_transpose not implemented"


ad.primitive_transposes[f_p] = f_jvp_transpose


k = 5
print("\nf (scalar)")
out = f(t_eval[k], inputs)
print(out.shape, " ", out)

print("\nf (vector)")
out = f(t_eval, inputs)
print(out.shape, " ", out)

print("\nf (vmap)")
out = jax.vmap(f, in_axes=(0, None))(t_eval, inputs)
print(out.shape, " ", out)

#print("\njvp (scalar) 1")
#flattree, treedef = tree_flatten(inputs)
#out = jax.jvp(f, (t_eval[k], inputs), (t_eval[k], inputs))
#print(out)

print("\njacfwd (scalar)")  # SAME VALUES FOR EACH INPUT
out = jax.jacfwd(f, argnums=1)(t_eval[k], inputs)
print(out)

print("\njacrev (scalar)")  # SAME VALUES FOR EACH INPUT
out = jax.jacrev(f, argnums=1)(t_eval[k], inputs)
print(out)

exit(0)

print("\njacfwd (scalar)")  # SAME VALUES FOR EACH INPUT
out = jax.jacfwd(f, argnums=1)(t_eval[k], inputs)
print(out)

print("\njacrev (scalar)")  # SAME VALUES FOR EACH INPUT
out = jax.jacrev(f, argnums=1)(t_eval[k], inputs)
print(out)

print("\ngrad (scalar)")  # ALL ZEROS
outvar = output_variables[0]
out = jax.grad(
    isolate_var(f, outvar),
    argnums=1
)(t_eval[k], inputs)
print(out)
