import time

import jax
import jax.numpy as jnp
import numpy as np

from core import (paulis, get_random_state, get_zero_state, state,
                  apply_unitary, unitary, exp_unitary, observable, expval)


def example_1():
    """Apply random Paulis to a random initial state"""
    n = 24
    ngates = 1000
    seed = 100
    np.random.seed(seed)
    phi = state(n, get_random_state(n))
    random_locations = list(map(lambda x: int(x), np.random.randint(0, n, ngates)))
    random_gates = list(map(lambda x: int(x), np.random.randint(0, 2, ngates)))
    pauli_names = ['X', 'Y', 'Z']
    gate_list = []
    # Perform  U |Psi>
    for i in range(ngates):
        if i == 1:
            print("Compiled contraction, starting timing...")
            start = time.time()
        u = unitary(random_locations[i], paulis[pauli_names[random_gates[i]]])
        gate_list.append((i, pauli_names[random_gates[i]]))
        apply_unitary(phi, u)
    print(f"Total time for applying {ngates} random gates to {n}-qubit state: {time.time() - start}")
    print(f"Gate list: {gate_list}")
    # Calculate <Psi|U^dag sum_i O_i U |Psi>
    energy = 0
    for i in range(n - 1):
        op = observable((i, i + 1), jnp.kron(paulis['Z'], paulis['Z']))
        energy += expval(phi, op)
    print(f"Energy {energy}")


def example_2():
    """Apply parameterized Unitaries to the zero state"""
    n = 24
    ngates = 1000
    seed = 100
    np.random.seed(seed)
    phi = state(n, get_zero_state(n))
    random_locations = list(map(lambda x: int(x), np.random.randint(0, n, ngates)))
    random_parameters = np.random.rand(ngates)
    random_gates = list(map(lambda x: int(x), np.random.randint(0, 2, ngates)))
    pauli_names = ['X', 'Y', 'Z']
    gate_list = []
    # Perform  U |Psi>
    for i in range(ngates):
        if i == 1:
            print("Compiled contraction, starting timing...")
            start = time.time()
        u = exp_unitary(random_locations[i], paulis[pauli_names[random_gates[i]]])
        gate_list.append((i, pauli_names[random_gates[i]]))
        apply_unitary(phi, u, random_parameters[i])
    print(f"Total time for applying {ngates} parameterized gates to {n}-qubit state: {time.time() - start}")
    print(f"Gate list: {gate_list}")
    # Calculate <Psi|U^dag sum_i O_i U |Psi>
    energy = 0
    for i in range(n - 1):
        op = observable((i, i + 1), jnp.kron(paulis['Z'], paulis['Z']))
        energy += expval(phi, op)
    print(f"Energy {energy}")


def example_3():
    """Apply parameterized Unitaries to the zero state and minimize energy via gradient"""
    n = 20
    ngates = 100
    seed = 100
    np.random.seed(seed)
    phi = state(n, get_zero_state(n))
    random_locations = list(map(lambda x: int(x), np.random.randint(0, n, ngates)))
    random_parameters = np.random.rand(ngates)
    random_gates = list(map(lambda x: int(x), np.random.randint(0, 2, ngates)))
    pauli_names = ['X', 'Y', 'Z']

    gate_list = []
    for i in range(ngates):
        gate_list.append(exp_unitary(random_locations[i], paulis[pauli_names[random_gates[i]]]))
    obs_list = []
    for i in range(n - 1):
        obs_list.append(observable((i, i + 1), jnp.kron(paulis['Z'], paulis['Z'])))

    # Construct loss function
    @jax.jit
    def loss_function(parameters):
        for i, u in enumerate(gate_list):
            apply_unitary(phi, u, parameters[i])
        energy = 0
        for op in obs_list:
            energy += expval(phi, op)
        return energy

    # Calculate <Psi|U^dag sum_i O_i U |Psi>
    print(f"VQE for {n}-qubit system with {ngates} gates")
    print(f"Energy {loss_function(random_parameters)}")
    grad_fn = jax.grad(loss_function)
    print(f"Initial loss:{loss_function(random_parameters)}")
    parameters = random_parameters.copy()
    lr = 0.01
    for step in range(100):
        gradient = grad_fn(parameters)
        parameters = parameters - lr * gradient
        print(f"Cost {loss_function(parameters)}")
        if jnp.mean(jnp.abs(gradient)) < 1e-3:
            print("Gradient magnitude smaller than 1e-3, stopping...")
            break


if __name__ == '__main__':
    example_1()
    example_2()
    example_3()
