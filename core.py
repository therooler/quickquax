from functools import partial
from typing import Tuple, Iterable, Any, Union

import flax.linen as nn
import jax
from flax import struct
from jax import numpy as jnp

paulis = {'I': jnp.eye(2).astype(complex),
          'X': jnp.array([[0, 1],
                          [1, 0]], complex),
          'Y': jnp.array([[0, -1j],
                          [1j, 0]], complex),
          'Z': jnp.array([[1, 0],
                          [0, -1]], complex)}

hadamard = jnp.array([[1, 1],
                      [1, -1]], complex) / jnp.sqrt(2)


def get_zero_state(n: int) -> jax.Array:
    phi = jnp.zeros(2 ** n, complex)
    phi = phi.at[0].set(1)
    return phi.reshape((2,) * n)


def get_random_state(n: int, seed: Union[int, jax.random.PRNGKey] = 0) -> jax.Array:
    if isinstance(seed, int):
        seed = jax.random.PRNGKey(seed=seed)
    elif not isinstance(seed, jax.random.PRNGKey):
        raise ValueError("`seed` must be an integer or jax.random.PRNGKey")
    phi = jax.random.normal(seed, shape=2 ** n, dtype=complex)
    phi = phi / jnp.linalg.norm(phi)
    return phi.reshape((2,) * n)


class State(nn.Module):
    n: int
    array: jax.Array


@struct.dataclass()
class Observable:
    loc: Tuple[int]
    array: jax.Array


@struct.dataclass()
class Unitary:
    loc: Tuple[int]
    array: jax.Array

    def __call__(self, *args):
        if args:
            raise ValueError(f'Did not expect arguments for unitary {self}')
        return self.array


@struct.dataclass()
class ExpUnitary(Unitary):
    loc: Tuple[int]
    array: jax.Array

    def __call__(self, *args):
        try:
            t = args[0]
        except IndexError:
            raise ValueError(f'Expected single argument for exponential, received {args}')
        return jax.scipy.linalg.expm(1j * self.array * t).reshape((2,) * 2 * len(self.loc))


def state(n: int, array: Any):
    assert isinstance(n, int), '`n` must be an integer'
    assert array.size == 2 ** n, '`array` must be have 2**n elements'
    array = jnp.array(array, dtype=complex).reshape((2,) * n)
    return State(n, array)


def observable(loc: Union[int, Tuple[int]], array: Any):
    loc = _check_loc(loc)
    loc = tuple(loc)
    nq = len(loc)
    assert array.size == 4 ** nq, '`array` must be have 4**nq elements'
    array = jnp.array(array, dtype=complex).reshape((2,) * 2 * nq)
    return Observable(loc, array)


def unitary(loc: Union[int, Tuple[int]], array: Any):
    loc = _check_loc(loc)
    loc = tuple(loc)
    nq = len(loc)
    assert array.size == 4 ** nq, '`array` must be have 4**nq elements'
    assert jnp.allclose(array @ array.conj().T, jnp.eye(2 ** nq)), '`array` must be unitary.'
    array = jnp.array(array, dtype=complex).reshape((2,) * 2 * nq)
    return Unitary(loc, array)


def exp_unitary(loc, array: Any):
    loc = _check_loc(loc)
    loc = tuple(loc)
    nq = len(loc)
    assert array.size == 4 ** nq, '`array` must be have 4**nq elements'
    assert jnp.allclose(array, array.conj().T), '`array` must be Hermitian.'
    array = jnp.array(array, dtype=complex)
    return ExpUnitary(loc, array)


def _check_loc(loc):
    assert isinstance(loc, Iterable) or isinstance(loc, int), '`loc` must be an iterable or integer'
    if isinstance(loc, int):
        loc = (loc,)
    assert isinstance(loc, Iterable), '`loc` must be an iterable'
    assert all(isinstance(i, int) for i in loc), 'All entries of loc must be integers'
    return loc


def indices_op_state(loc: Tuple[int], n: int):
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


def expval(state: State, op: Observable):
    idx_op, idx_state, idx_out = indices_op_state(op.loc, state.n)
    op_state = contract(op.array, state.array, idx_op, idx_state, idx_out)
    idx_state = tuple(range(state.n))
    return contract(op_state, state.array.conj(), idx_state, idx_state, ()).real


def apply_unitary(state: State, unitary: Unitary, *args):
    idx_op, idx_state, idx_out = indices_op_state(unitary.loc, state.n)
    state.array = contract(unitary(*args), state.array, idx_op, idx_state, idx_out)


@partial(jax.jit, static_argnums=(2, 3, 4))
def contract(a: jax.Array, b: jax.Array, idx_a: Tuple[int], idx_b: Tuple[int], idx_out: Tuple[int]):
    return jnp.einsum(a, idx_a, b, idx_b, idx_out)
