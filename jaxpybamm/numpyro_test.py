import jax.numpy as jnp
from jax import custom_vjp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS
from jax import vmap


def test_single_input():
    # Custom Jax primitive
    @custom_vjp
    def h(x):
        return jnp.sin(x)

    def h_fwd(x):
        return h(x), jnp.cos(x)

    def h_bwd(res, u):
        cos_x = res
        return (cos_x * u,)

    h.defvjp(h_fwd, h_bwd)

    # Generate test data
    np.random.seed(32)
    sigin = 0.3
    N = 20
    x = np.sort(np.random.rand(N)) * 4 * np.pi
    data = h(x) + np.random.normal(0, sigin, size=N)

    # Model
    def model(x, y):
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        x0 = numpyro.sample("x0", dist.Uniform(-1.0, 1.0))
        mu = h(x - x0)
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    # Inference
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    num_warmup, num_samples = 1000, 2000
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
    mcmc.run(rng_key_, x=x, y=data)
    mcmc.print_summary()


def test_multi_input():
    # Custom Jax primitive
    @custom_vjp
    def h(x, A):
        return A * jnp.sin(x)

    def h_fwd(x, A):
        res = (A * jnp.cos(x), jnp.sin(x))
        return h(x, A), res

    def h_bwd(res, u):
        A_cos_x, sin_x = res
        return (A_cos_x * u, sin_x * u)

    h.defvjp(h_fwd, h_bwd)

    # Generate test data
    np.random.seed(32)
    sigin = 0.3
    N = 20
    x = np.sort(np.random.rand(N)) * 4 * np.pi
    A = 0.1 + 0.0 * np.random.normal(1.75, 0.1, size=N)
    data = h(x, A) + np.random.normal(0, sigin, size=N)

    # Model
    def model(x, y):
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        x0 = numpyro.sample("x0", dist.Uniform(-1.0, 1.0))
        A = numpyro.sample("A", dist.Exponential(1.0))
        hv = vmap(h, (0, None), 0)
        mu = hv(x - x0, A)
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    # Inference
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    num_warmup, num_samples = 1000, 2000
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
    mcmc.run(rng_key_, x=x, y=data)
    mcmc.print_summary()


if __name__ == "__main__":
    test_single_input()
    test_multi_input()
