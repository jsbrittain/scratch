import jax
import numpy as onp
from jax.interpreters import ad
from jax import lax

# Make a Primitive
foo_p = jax.core.Primitive("foo")


def foo(x1, x2):
    return foo_p.bind(x1, x2)


# Give it an impl rule, which just accepts as input an raw array-like value (no
# tracers or anything) and applies opaque foreign functions to produce a raw
# array-like value output.
@foo_p.def_impl
def foo_impl(x1, x2):
    return x1 * x2


# Give it a JVP rule, which trivially just calls another primitive we'll define.
def foo_jvp(primals, tangents):
    x1, x2 = primals
    x1dot, x2dot = tangents
    y = foo(x1, x2)

    def make_zero(tan):
        return lax.zeros_like_array(x1) if type(tan) is ad.Zero else tan

    y_dot = foo_jvp_p.bind(
        x1,
        x2,
        make_zero(x1dot),
        make_zero(x2dot),
    )
    return y, y_dot


ad.primitive_jvps[foo_p] = foo_jvp

foo_jvp_p = jax.core.Primitive("foo_jvp")


# We could define an impl rule for foo_jvp_p, and thus get first-order
# forward-mode AD working. But if we only care about reverse-mode, we actually
# don't need one; instead we need an abstract eval rule and a transpose rule.
# The transpose rule can do the full VJP calculation, and can itself call an
# opaque primitive.
@foo_jvp_p.def_abstract_eval
def foo_jvp_abstract_eval(x1_aval, x2_aval, x1_dot_aval, x2_dot_aval):
    y_dot_aval = jax.core.ShapedArray(x1_dot_aval.shape, x1_dot_aval.dtype)
    return y_dot_aval


def foo_jvp_transpose(y_bar, *args):  # x1, x2, x1dot_dummy, x2dot_dummy):
    # print("foo_jvp_transpose: ", y_bar, x1, x2, x1dot_dummy, x2dot_dummy)
    x1, x2, x1dot_dummy, x2dot_dummy = args
    if ad.is_undefined_primal(x1dot_dummy):
        # x_bar = foo_vjp_p.bind(x1, x2, y_bar)  # y_bar aka y_grad
        return None, None, x2, None
    if ad.is_undefined_primal(x2dot_dummy):
        # x_bar = foo_vjp_p.bind(x1, x2, y_bar)  # y_bar aka y_grad
        return None, None, None, x1


ad.primitive_transposes[foo_jvp_p] = foo_jvp_transpose


# Finally, let's write the vjp rule as a primitive.
foo_vjp_p = jax.core.Primitive("foo_vjp")


# @foo_vjp_p.def_impl
# def foo_vjp_impl(x1, x2, y_bar):
#     print("foo_vjp_impl: ", x1, x2, y_bar)
#     return onp.cos(x1) + 2 * x2


# Let's test it!
print(jax.grad(foo, argnums=0)(3.0, 1.0))
print(jax.grad(foo, argnums=1)(1.0, 3.0))

print(jax.grad(foo, argnums=0)(1.0, 3.0))
print(jax.grad(foo, argnums=1)(3.0, 1.0))
