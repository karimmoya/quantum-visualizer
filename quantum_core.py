import random
from typing import List, Dict, Any
import numpy as np


IDENTITY = np.array([[1, 0], [0, 1]], dtype=np.complex128)
H_GATE = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
X_GATE = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=np.complex128)

P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)

SINGLE_QUBIT_GATES = {
    "h": H_GATE,
    "x": X_GATE,
    "y": Y_GATE,
    "z": Z_GATE
}


def _combine_operators(operations_list: List[np.ndarray]) -> np.ndarray:
    reversed_ops = operations_list[::-1]

    full_operator = reversed_ops[0]
    for op in reversed_ops[1:]:
        full_operator = np.kron(full_operator, op)

    return full_operator


def _build_single_qubit_operator(num_qubits: int, target_qubit: int, gate_matrix: np.ndarray) -> np.ndarray:
    ops_list = []
    for i in range(num_qubits):
        if i == target_qubit:
            ops_list.append(gate_matrix)
        else:
            ops_list.append(IDENTITY)

    return _combine_operators(ops_list)


def _build_cnot_operator(num_qubits: int, control: int, target: int) -> np.ndarray:
    ops_zero = []
    ops_one = []

    for i in range(num_qubits):
        if i == control:
            ops_zero.append(P0)
            ops_one.append(P1)
        elif i == target:
            ops_zero.append(IDENTITY)
            ops_one.append(X_GATE)
        else:
            ops_zero.append(IDENTITY)
            ops_one.append(IDENTITY)

    operator_zero = _combine_operators(ops_zero)
    operator_one = _combine_operators(ops_one)

    return operator_zero + operator_one


def calculate_probabilities(state_vector: np.ndarray) -> np.ndarray:
    return np.abs(state_vector) ** 2


def measure_shot(probabilities: np.ndarray) -> int:
    random_val = random.random()
    cumulative = 0.0

    for i, p in enumerate(probabilities):
        cumulative += p
        if random_val < cumulative:
            return i

    return len(probabilities) - 1  # Fallback seguro (no debería ocurrir)


def execute_circuit(num_qubits: int, operations: List[Dict[str, Any]]) -> np.ndarray:
    dim = 2 ** num_qubits
    current_state = np.zeros(dim, dtype=np.complex128)
    current_state[0] = 1.0 + 0j

    for op in operations:
        gate_type = op["gate"]
        operator = None

        if gate_type in SINGLE_QUBIT_GATES:
            operator = _build_single_qubit_operator(
                num_qubits,
                op["target"],
                SINGLE_QUBIT_GATES[gate_type]
            )

        elif gate_type == "cx":
            operator = _build_cnot_operator(
                num_qubits,
                op["control"],
                op["target"]
            )

        if operator is not None:
            current_state = np.dot(operator, current_state)
        else:
            print(f"Warning: Operación '{gate_type}' no soportada o inválida.")

    return current_state


# --- TESTING ---
if __name__ == "__main__":
    test_ops = [
        {'gate': 'h', 'target': 0},
        {'gate': 'cx', 'control': 0, 'target': 1}
    ]

    final_state = execute_circuit(num_qubits=2, operations=test_ops)
    probs = calculate_probabilities(final_state)
    result = measure_shot(probs)

    print(f"Estado Final: {final_state}")
    print(f"Probabilidades: {probs}")
    print(f"Resultado Medición: {result}")