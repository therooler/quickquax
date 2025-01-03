# Quantum State Simulator in Jax

To run this code, create a virtual environment with Python 3.12 and install the requirements.txt.
The only dependencies are `jax`, `flax` and `pytest`.

To test the code, activate your environment and run

```bash
cd tests
pytest .
```

which should be all green.

## Examples

There are three example to illustrate the (bare) functionality of the code.
- Example 1 applies 1000 random Pauli gates to a 24-qubit state and calculates the energy of a simple Hamiltonian.
- Example 2 applies 1000 random Parameterized gates to a 24-qubit state and calculates the energy of a simple Hamiltonian.
On a 2022 MacM1, examples 1 and 2 take 20 seconds, consuming no more than 200mb of memory.
- Example 3 applies 100 random Pauli gates to a 20-qubit state and minimizes the energy of a simple Hamiltonian via gradient descent.
Example 3 takes a little longer to run, but should finish within a couple of minutes.

To run the examples, run

```bash
python examples
```

