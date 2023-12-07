from jax import jvp

def f(z, x, y):
    return x['hi'] + y['foo']


f(0.1, {'hi': 1.}, {'foo': 2.})

print(
    jvp(f, [0.1, {'hi': 1.}, {'foo': 2.}], [0.2, {'hi': 3.}, {'foo': 4.}])
)
