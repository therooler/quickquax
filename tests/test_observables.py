import sys

sys.path.append('..')

import jax.numpy as jnp
import pytest

from core import paulis, get_random_state, state, observable, expval


@pytest.mark.parametrize('pauli', ['I', 'X', 'Y', 'Z'])
@pytest.mark.parametrize('loc', [0, 1, ])
def test_single_paulis_2q(loc, pauli):
    n = 2
    phi = state(n, get_random_state(n))
    phi_true = phi.array.copy()
    obs0 = observable((loc,), paulis[pauli])
    val = expval(phi, obs0)
    if loc == 0:
        val_true = phi_true.conj().flatten() @ jnp.kron(paulis[pauli], paulis['I']) @ phi_true.flatten()
    else:
        val_true = phi_true.conj().flatten() @ jnp.kron(paulis['I'], paulis[pauli]) @ phi_true.flatten()
    assert jnp.allclose(val, val_true)


@pytest.mark.parametrize('pauli_0', ['I', 'X', 'Y', 'Z'])
@pytest.mark.parametrize('pauli_1', ['I', 'X', 'Y', 'Z'])
def test_double_paulis_2q(pauli_0, pauli_1):
    n = 2
    phi = state(n, get_random_state(n))
    phi_true = phi.array.copy()
    op = jnp.kron(paulis[pauli_0], paulis[pauli_1])
    obs0 = observable((0, 1), op)
    val = expval(phi, obs0)
    val_true = phi_true.conj().flatten() @ op @ phi_true.flatten()
    assert jnp.allclose(val, val_true)


@pytest.mark.parametrize('pauli_0', ['I', 'X', 'Y', 'Z'])
@pytest.mark.parametrize('pauli_1', ['I', 'X', 'Y', 'Z'])
def test_double_paulis_3q(pauli_0, pauli_1):
    n = 3
    phi = state(n, get_random_state(n))
    phi_true = phi.array.copy()
    op = jnp.kron(paulis[pauli_0], paulis[pauli_1])
    op_true = jnp.kron(paulis[pauli_0], jnp.kron(jnp.eye(2, ), paulis[pauli_1]))
    obs0 = observable((0, 2), op)
    val = expval(phi, obs0)
    val_true = phi_true.conj().flatten() @ op_true @ phi_true.flatten()
    assert jnp.allclose(val, val_true)
