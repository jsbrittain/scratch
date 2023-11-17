import jax
import numpy as onp
from jax.interpreters import ad
from jax.interpreters import mlir
from jax._src.lib.mlir.dialects import hlo

# Make a Primitive
foo_p = jax.core.Primitive("foo")


def foo(x1):
    return foo_p.bind(x1)


# Give it an impl rule, which just accepts as input an raw array-like value (no
# tracers or anything) and applies opaque foreign functions to produce a raw
# array-like value output.
@foo_p.def_impl
def foo_impl(x1):
    # Simulate a model that takes a vector of inputs and outputs a single value
    return onp.sin(x1)


@foo_p.def_abstract_eval
def foo_abstract_eval(x1_aval):
    return jax.core.ShapedArray(x1_aval.shape, x1_aval.dtype)


def foo_lowering(ctx, x1c):
    """The compilation to XLA of the primitive.

    Given an mlir.ir.Value for each argument, return the mlir.ir.Values for
    the results of the function.

    Does NOT need to be a JAX-traceable function.
    """
    print("lowering: ", ctx, x1c)
    return [hlo.SineOp(x1c).result]


mlir.register_lowering(foo_p, foo_lowering, platform="cpu")


# Give it a JVP rule, which trivially just calls another primitive we'll define.
def foo_jvp(primals, tangents):
    (x1,), (x1dot,) = primals, tangents
    y = foo(x1)
    y_dot = foo_jvp_p.bind(x1, x1dot)
    return y, y_dot


ad.primitive_jvps[foo_p] = foo_jvp

foo_jvp_p = jax.core.Primitive("foo_jvp")


# We could define an impl rule for foo_jvp_p, and thus get first-order
# forward-mode AD working. But if we only care about reverse-mode, we actually
# don't need one; instead we need an abstract eval rule and a transpose rule.
# The transpose rule can do the full VJP calculation, and can itself call an
# opaque primitive.
@foo_jvp_p.def_abstract_eval
def foo_jvp_abstract_eval(x_aval, x_dot_aval):
    y_dot_aval = jax.core.ShapedArray(x_dot_aval.shape, x_dot_aval.dtype)
    return y_dot_aval


def foo_jvp_transpose(y_bar, x, x_dot_dummy):
    assert ad.is_undefined_primal(x_dot_dummy)  # just a dummy input
    x_bar = foo_vjp_p.bind(x, y_bar)  # y_bar aka y_grad
    return None, x_bar  # None for nonlinear primal input x


ad.primitive_transposes[foo_jvp_p] = foo_jvp_transpose

# Finally, let's write the vjp rule as a primitive.
foo_vjp_p = jax.core.Primitive("foo_vjp")


@foo_vjp_p.def_impl
def foo_vjp_impl(x, y_bar):
    return y_bar * onp.cos(onp.sin(x)) * onp.cos(x)


###


# Let's test it!
d = onp.array([3.0, 1.0, 2.0])
print("direct: ", foo(d))
print("jit:    ", jax.jit(foo)(d))
print("grad:   ", jax.vmap(jax.grad(foo))(d))
