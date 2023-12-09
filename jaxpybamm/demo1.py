from jax import jvp


def f(z, x, y):
    return x["hi"] + y["foo"]


f(0.1, {"hi": 1.0}, {"foo": 2.0})

print(jvp(f, [0.1, {"hi": 1.0}, {"foo": 2.0}], [0.2, {"hi": 3.0}, {"foo": 4.0}]))
