import time
import jax
import jax.numpy as jnp
from core import (paulis, get_zero_state, _check_loc)
from functools import partial

ZZ = jnp.kron(paulis['Z'], paulis['Z']).reshape([2] * 4) * jnp.pi / 2
X = paulis['X']
Y = paulis['Y']
from typing import Tuple


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
    return jax.jit(lambda x, u: contract(op1 * u[0] + op2* u[1], x, idx_op, idx_state, idx_out))


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
        fns.append(get_apply_add_fn_u(X,Y, (i,), n))
    return fns


def get_time_step(dt, n):
    """Apply Hamiltonian"""
    fns0 = apply_H0(dt, n)
    fnsI = apply_HI(n)

    @jax.jit
    def time_step(_phi, _u):
        new_phi = jnp.zeros_like(_phi)
        for f in fns0:
            new_phi += f(_phi)
        for f, _u_i in zip(fnsI, _u):
            new_phi += f(_phi, _u_i)
        return _phi + new_phi

    return time_step


def main(key, n, steps=100, dt=1e-3):
    array = get_zero_state(n)
    phi = jnp.array(array, dtype=complex).reshape((2,) * n)
    u = jnp.ones((n, 2))
    dW = jax.random.normal(key, (steps, n, 2)) * jnp.sqrt(dt)
    time_step = get_time_step(dt, n)
    for i, s in enumerate(range(steps)):
        if i == 1:
            start = time.time()
        phi = time_step(phi, u*dt + dW[s])
        phi / jnp.linalg.norm(phi.flatten())
    print(f"time: {time.time() - start}")


if __name__ == '__main__':
    # n = 10, 0.02s per 100 steps.
    steps = 100
    master_key = jax.random.PRNGKey(1000)
    main(master_key, n=10, steps = steps)

    main_partial = partial(main, n=10,steps=steps)
    keys = jax.random.split(master_key, steps)

    # print(keys)
    # jax.vmap(partial(main, n=12, steps=100, dt=1e-2))(keys)
