import jax
import numpy as onp
from jax.interpreters import ad

f_p = jax.core.Primitive('f')


def f(x):
    print("f", x)
    return f_p.bind(x)


@f_p.def_impl
def f_impl(x_arr):
    print("f_impl", x_arr)
    y = onp.sin(onp.sin(x_arr))
    return y


def f_jvp(primals, tangents):
    print("f_jvp", primals, tangents)
    x, = primals
    x_dot, = tangents
    y = f(x)
    y_dot = f_jvp_p.bind(x, x_dot)
    return y, y_dot


ad.primitive_jvps[f_p] = f_jvp
f_jvp_p = jax.core.Primitive('f_jvp')


@f_jvp_p.def_abstract_eval
def f_jvp_abstract_eval(x_aval, x_dot_aval):
    print("f_jvp_abstract_eval", x_aval, x_dot_aval)
    y_dot_aval = jax.core.ShapedArray(x_dot_aval.shape, x_dot_aval.dtype)
    return y_dot_aval


def f_jvp_transpose(y_bar, x, x_dot_dummy):
    print("f_jvp_transpose", y_bar, x, x_dot_dummy)
    assert ad.is_undefined_primal(x_dot_dummy)
    x_bar = f_jvp_p.bind(x, y_bar)
    return None, x_bar


ad.primitive_transposes[f_jvp_p] = f_jvp_transpose
f_vjp_p = jax.core.Primitive('f_vjp')


@f_jvp_p.def_impl
def f_vjp_impl(x, y_bar):
    print("f_vjp_impl", x, y_bar)
    return y_bar * onp.cos(onp.sin(x)) * onp.cos(x)


#print(jax.jit(f))
#print(f(1.5))
print(jax.grad(f)(1.5))
#
#print(jax.jacfwd(f)(1.5))
#print(jax.jacrev(f)(1.5))
#
#primals, tangents = jax.jvp(f, (0.1,), (0.2,))
#print(primals, tangents)
#
#primals, tangents = jax.vjp(f, 0.1)
#print(primals, tangents)
