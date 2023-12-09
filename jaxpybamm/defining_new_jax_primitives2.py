import jax
import numpy as onp
import jax.numpy as jnp
from jax.interpreters import ad

# Make a Primitive
foo_p = jax.core.Primitive("foo")


def foo(x):
    return foo_p.bind(x)


# Give it an impl rule, which just accepts as input an raw array-like value (no
# tracers or anything) and applies opaque foreign functions to produce a raw
# array-like value output.
@foo_p.def_impl
def foo_impl(x_arr):
    return onp.sin(onp.sin(x_arr))


# Give it a JVP rule, which trivially just calls another primitive we'll define.
def foo_jvp(primals, tangents):
    (x,), (xdot,) = primals, tangents
    y = foo(x)
    y_dot = foo_jvp_p.bind(x, xdot)
    return y, y_dot


ad.primitive_jvps[foo_p] = foo_jvp

foo_jvp_p = jax.core.Primitive("foo_jvp")


def f_batch(args, batch_axes):
    print("f_batch")
    print("  args: ", args)
    print("  batch_axes: ", batch_axes)
    t = args[0]
    if batch_axes[0] is not None:
        out = list(map(foo, t))
    else:
        raise Exception("Not implemented")
    return jnp.stack(out), 0


jax.interpreters.batching.primitive_batchers[foo_p] = f_batch


# We could define an impl rule for foo_jvp_p, and thus get first-order
# forward-mode AD working. But if we only care about reverse-mode, we actually
# don't need one; instead we need an abstract eval rule and a transpose rule.
# The transpose rule can do the full VJP calculation, and can itself call an
# opaque primitive.
@foo_jvp_p.def_abstract_eval
def foo_jvp_abstract_eval(x_aval, x_dot_aval):
    print("foo_jvp_abstract_eval")
    print("  x_aval: ", x_aval)
    print("  x_dot_aval: ", x_dot_aval)
    y_dot_aval = jax.core.ShapedArray(x_dot_aval.shape, x_dot_aval.dtype)
    return y_dot_aval


@foo_jvp_p.def_impl
def foo_jvp_impl(x, x_dot):
    return onp.cos(onp.sin(x)) * onp.cos(x)


def foo_jvp_batch(args, batch_axes):
    print("f_batch")
    print("  args: ", args)
    print("  batch_axes: ", batch_axes)
    out = []
    if batch_axes[0] is not None:
        for a in args[0]:
            _, outa = foo_jvp((a,), args[1:])
            out.append(outa)
    elif batch_axes[1] is not None:
        for a in args[1]:
            _, outa = foo_jvp((args[0],), (a,))
            out.append(outa)
    else:
        print("batch_axes = ", batch_axes)
        raise Exception("not implemented")
    return jnp.stack(out), 0


jax.interpreters.batching.primitive_batchers[foo_jvp_p] = foo_jvp_batch


def foo_jvp_transpose(y_bar, x, x_dot_dummy):
    assert ad.is_undefined_primal(x_dot_dummy)  # just a dummy input
    x_bar = foo_vjp_p.bind(x, y_bar)  # y_bar aka y_grad
    print(x_bar)
    return None, x_bar  # None for nonlinear primal input x


ad.primitive_transposes[foo_jvp_p] = foo_jvp_transpose

# Finally, let's write the vjp rule as a primitive.
foo_vjp_p = jax.core.Primitive("foo_vjp")


def foo_vjp(x):
    return foo_vjp_p.bind(x)


@foo_vjp_p.def_impl
def foo_vjp_impl(x, y_bar):
    return y_bar * onp.cos(onp.sin(x)) * onp.cos(x)


def foo_vjp_batch(args, batch_axes):
    print("foo_vjp_batch")
    print("  args:", args)
    print("  batch_axes:", batch_axes)

    x, y_bar = args
    out = []
    if batch_axes[0] is not None:
        # batch over time
        out = list(map(lambda x1: foo_vjp_p.bind(x1, y_bar), x))
    elif batch_axes[1] is not None:
        # batch over tangents
        for y in y_bar:
            outy = foo_vjp_p.bind(x, y)
            out.append(outy)
    else:
        raise ValueError("must be at least one non-None batch axis")
    return jnp.stack(out), 0


jax.interpreters.batching.primitive_batchers[foo_vjp_p] = foo_vjp_batch


###


# Let's test it!

print("\n\njacfwd")
print(jax.jacfwd(foo)(3.0))

print("\n\njacfwd vmap")
print(jax.vmap(jax.jacfwd(foo))(onp.arange(20, dtype="float64")))

print("\n\njacrev")
print(jax.jacrev(foo)(3.0))

print("\n\njacrev vmap")
print(jax.vmap(jax.jacrev(foo))(onp.arange(20, dtype="float64")))

print("\n\ngrad")
print(jax.grad(foo)(3.0))

print("\n\ngrad vmap")
print(jax.vmap(jax.grad(foo))(onp.arange(20, dtype="float64")))
