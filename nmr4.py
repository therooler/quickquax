import time
import jax
import jax.numpy as jnp
from core import (paulis, get_zero_state, _check_loc)
from functools import partial

from typing import Tuple
import numpy as np

ZZ = jnp.kron(paulis['Z'], paulis['Z']).reshape([2] * 4) * jnp.pi / 2
X = paulis['X']
Y = paulis['Y']


def indices_op_state(loc: Tuple[int], n: int):
    loc = _check_loc(loc)
    nq = len(loc)
    idx_op = tuple(range(2 * nq))
    idx_out = list(range(2 * n, 2 * n + n))
    idx_state = idx_out.copy()
    for i_l, l in enumerate(loc):
        idx_state[l] = idx_op[i_l + nq]
        idx_out[l] = idx_op[i_l]
    idx_state = tuple(idx_state)
    idx_out = tuple(idx_out)
    return idx_op, idx_state, idx_out


def get_apply_add_fn(operator, loc, n: int):
    idx_op, idx_state, idx_out = indices_op_state(loc, n)
    return jax.jit(lambda x: contract(operator, x, idx_op, idx_state, idx_out))


def get_apply_add_fn_u(op1, op2, loc, n: int):
    idx_op, idx_state, idx_out = indices_op_state(loc, n)
    return jax.jit(lambda x, u: contract(op1 * u[0] + op2 * u[1], x, idx_op, idx_state, idx_out))


@partial(jax.jit, static_argnums=(2, 3, 4))
def contract(a: jax.Array, b: jax.Array, idx_a: Tuple[int], idx_b: Tuple[int], idx_out: Tuple[int]):
    return jnp.einsum(a, idx_a, b, idx_b, idx_out)


def apply_H0(dt, n):
    fns = []
    for i in range(n):
        for j in range(i + 1, n):
            fns.append(get_apply_add_fn(ZZ * dt, (i, j), n))
    return fns


def apply_HI(n):
    fns = []
    for i in range(n):
        fns.append(get_apply_add_fn_u(X, Y, (i,), n))
    return fns


def get_time_step(dt, n):
    """Apply Hamiltonian"""
    fns0 = apply_H0(dt, n)
    fnsI = apply_HI(n)
    index0 = jnp.arange(len(fns0))
    indexI = jnp.arange(len(fnsI))
    print(len(index0), len(indexI))
    vmap_functions0 = jax.vmap(lambda i, x: jax.lax.switch(i, fns0, x), in_axes=0)
    vmap_functionsI = jax.vmap(lambda i, x: jax.lax.switch(i, fnsI, *x), in_axes=0)

    @jax.jit
    def time_step(_phi, _u):
        new_phis0 = vmap_functions0(index0, jnp.stack([_phi] * len(index0)))
        new_phisI = vmap_functionsI(indexI, (jnp.stack([_phi] * len(indexI)), _u))
        return _phi + jnp.sum(new_phis0, axis=0) + jnp.sum(new_phisI, axis=0)

    return time_step


def main(master_key, n, steps=100, dt=1e-3, ntraj: int = 1):
    time_step = get_time_step(dt, n)

    @jax.jit
    def single_trajectory(key):
        array = get_zero_state(n)
        phi = jnp.array(array, dtype=complex).reshape((2,) * n)
        u = jnp.ones((n, 2)) * dt
        dW = jax.random.normal(key, (steps, n, 2)) * jnp.sqrt(dt)
        for i, s in enumerate(range(steps)):
            phi = time_step(phi, u + dW[s])
            phi /= jnp.linalg.norm(phi)
        return phi

    @jax.jit
    def get_trajectories(key):
        keys = jax.random.split(key, ntraj)
        return jax.vmap(single_trajectory)(keys)

    for l in range(10):
        start = time.time()
        master_key, trajectory_key = jax.random.split(master_key, 2)
        print(f"Getting trajectory with key {master_key}")
        single_trajectory(trajectory_key)
        # phis = get_trajectories(trajectory_key)
        # print(np.array(phis).shape)
        print(f"time for {ntraj} trajectories - n={n}: {time.time() - start}")
    # 9.009130954742432 seconds for 1000 trajectories


if __name__ == '__main__':
    # n = 10, 0.02s per 100 steps.
    seed = 1000
    master_key = jax.random.PRNGKey(seed)
    main(master_key, n=10, ntraj=1)

    # print(keys)
    # jax.vmap(partial(main, n=12, steps=100, dt=1e-2))(keys)
