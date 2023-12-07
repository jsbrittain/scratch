import jax
import jax.numpy as jnp
import numpy as np


def f(t, x):
    return jnp.array([
        t,
        x[0],
        x[1] + 5 * x[2],
        t * x[0] * x[1] * x[2],
    ])


print(
    f(1., np.array([1., 2., 3.]))
)

print(
    jax.jacfwd(f, argnums=0)(1., np.array([1., 2., 3.]))
)

print(
    jax.jacfwd(f, argnums=1)(1., np.array([1., 2., 3.]))
)

print('\n\n')


def f2(t, x, y, z):
    return jnp.array([
        y + 5 * z,
        t * x * y * z,
    ])


print(
    f2(1., 1., 2., 3.)
)

print(
    jax.jacfwd(f2, argnums=0)(1., 1., 2., 3.)
)
print(
    jax.jacfwd(f2, argnums=1)(1., 1., 2., 3.)
)
print(
    jax.jacfwd(f2, argnums=2)(1., 1., 2., 3.)
)
print(
    jax.jacfwd(f2, argnums=3)(1., 1., 2., 3.)
)

print('\n\n')


def f3(t, x):
    return jnp.array([
        x['y'] + 5 * x['z'],
        t * x['x'] * x['y'] * x['z'],
    ])


x = {'x': 1.0, 'y': 2.0, 'z': 3.0}
print(
    f3(1., x)
)

print(
    jax.jacfwd(f3, argnums=0)(1., x)
)
print(
    jax.jacfwd(f3, argnums=1)(1., x)
)
