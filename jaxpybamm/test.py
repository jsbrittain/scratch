from jax import core
import numpy as np
from jax._src import api
from jax._src.lib.mlir.dialects import hlo


def multiply_add_impl(x, y, z):
    """concrete implementation
    does not need to be JAX traceable
    (demonstrated by using vanilla numpy functions, not jnp)
    """
    return np.add(np.multiply(x, y), z)


def multiply_add_abstract_eval(xs, ys, zs):
    """abstract evaluation of the primitive
    does not need to be jax traceable"""
    assert xs.shape == ys.shape
    assert xs.shape == zs.shape
    return core.ShapedArray(xs.shape, xs.dtype)


def multiply_add_lowering(ctx, xc, yc, zc):
    """Compilation to XLA of the primitive
    This could be any arbitrary C++ code using a CustomCall"""
    return [hlo.AddOp(hlo.MulOp(xc, yc), zc).result]


multiply_add_p = core.Primitive("multiply_add")  # create primitive
multiply_add_p.def_impl(multiply_add_impl)
multiply_add_p.def_abstract_eval(multiply_add_abstract_eval)

from jax.interpreters import mlir
mlir.register_lowering(multiply_add_p, multiply_add_lowering, platform='cpu')


def multiply_add_prim(x, y, z):
    """jax traceable implementation"""
    return multiply_add_p.bind(x, y, z)


def square_add_prim(a, b):
    return multiply_add_prim(a, a, b)


print(square_add_prim(2., 10.))
print(api.jit(square_add_prim)(2., 10.))
print(api.grad(square_add_prim, argnums=0)(2.0, 10.))
