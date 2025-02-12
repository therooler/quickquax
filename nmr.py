import time
import jax
import jax.numpy as jnp
from core import (paulis, get_zero_state, state, operator, apply_add)
from functools import partial

ZZ = jnp.kron(paulis['Z'], paulis['Z']).reshape([2]) * jnp.pi / 2
X = paulis['X']
Y = paulis['Y']


def apply_H0(phi, dt, n):
    for i in range(n):
        for j in range(i, n):
            # apply zi zj * pi/2
            op = operator((i, j), ZZ * dt)
            phi = apply_add(phi, op, n)
    return phi


def apply_HI(phi, u, n):
    for i in range(n):
        # apply u_i^x X + u_i^y Y
        op = operator((i,), u[i, 0] * X + u[i, 1] * Y)
        phi = apply_add(phi, op, n)
    return phi


def time_step(phi, u, dW, dt, n):
    """Apply Hamiltonian"""
    phi = apply_H0(phi, dt, n)
    phi = apply_HI(phi, u + dW, n)
    return phi


def main(key, n, steps=100, dt=1e-2):
    array = get_zero_state(n)
    phi = jnp.array(array, dtype=complex).reshape((2,) * n)
    u = jnp.ones((n, 2))
    dW = jax.random.normal(key, (steps, n, 2)) * jnp.sqrt(dt)
    for i, s in enumerate(range(steps)):
        if i == 1:
            start = time.time()
        phi = time_step(phi, u, dW[s], dt, n)
    print(f"time: {time.time() - start}")


if __name__ == '__main__':
    # n = 10, 0.02s per 100 steps.
    master_key = jax.random.PRNGKey(1000)
    main(master_key, n=10)

    # master_key, keys = jax.random.split(master_key, 2)
    # print(keys)
    # jax.vmap(partial(main, n=12, steps=100, dt=1e-2))(keys)
