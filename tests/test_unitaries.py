import sys

sys.path.append('..')

import jax
import jax.numpy as jnp
import pytest

from core import paulis, get_random_state, state, observable, expval, exp_unitary, unitary, apply_unitary


@pytest.mark.parametrize('pauli', ['I', 'X', 'Y', 'Z'])
@pytest.mark.parametrize('loc', [0, 1, ])
def test_unitary_paulis_2q(loc, pauli):
    n = 2
    phi = state(n, get_random_state(n))
    phi_true = phi.array.copy()
    op = unitary((loc,), paulis[pauli])
    if loc == 0:
        op_true = jnp.kron(paulis[pauli], paulis['I'])
        val_true = op_true @ phi_true.flatten()
    else:
        op_true = jnp.kron(paulis['I'], paulis[pauli])
        val_true = op_true @ phi_true.flatten()
    with pytest.raises(ValueError, match='Did not expect arguments for unitary'):
        apply_unitary(phi, op, 1.0)
    apply_unitary(phi, op)
    assert jnp.allclose(phi.array.flatten(), val_true.flatten())


@pytest.mark.parametrize('pauli', ['I', 'X', 'Y', 'Z'])
@pytest.mark.parametrize('loc', [0, 1, ])
def test_exponential_paulis_2q(loc, pauli):
    n = 2
    phi = state(n, get_random_state(n))
    phi_true = phi.array.copy()
    op = exp_unitary((loc,), paulis[pauli])
    if loc == 0:
        op_true = jax.scipy.linalg.expm(1j * jnp.kron(paulis[pauli], paulis['I']))
        val_true = op_true @ phi_true.flatten()
    else:
        op_true = jax.scipy.linalg.expm(1j * jnp.kron(paulis['I'], paulis[pauli]))
        val_true = op_true @ phi_true.flatten()
    apply_unitary(phi, op, 1.)
    assert jnp.allclose(phi.array.flatten(), val_true.flatten())


@pytest.mark.parametrize('pauli_0', ['I', 'X', 'Y', 'Z'])
@pytest.mark.parametrize('pauli_1', ['I', 'X', 'Y', 'Z'])
def test_double_paulis_3q(pauli_0, pauli_1):
    n = 3
    phi = state(n, get_random_state(n))
    op = jnp.kron(paulis[pauli_0], paulis[pauli_1])
    op_true = jnp.kron(paulis[pauli_0], jnp.kron(jnp.eye(2, ), paulis[pauli_1]))
    obs0 = observable((0, 2), op)
    val = expval(phi, obs0)
    val_true = phi.array.conj().flatten() @ op_true @ phi.array.flatten()
    assert jnp.allclose(val, val_true)
