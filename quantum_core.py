import random
import numpy as np


def _combine_operators(operations_list: list) -> np.ndarray:
    reversed_ops = list(reversed(operations_list))
    partial_result = reversed_ops[0]
    for op in reversed_ops[1:]:
        partial_result = np.kron(partial_result, op)
    return partial_result


def calculate_probabilities(current_state: np.ndarray) -> np.ndarray:
    absolute_value = np.absolute(current_state)
    probabilities = absolute_value ** 2
    return probabilities


def measure_shot(probabilities: np.ndarray) -> int:
    random_shot = random.random()
    cumulative_probability = 0.0
    for i, probability in enumerate(probabilities):
        cumulative_probability += probability
        if random_shot < cumulative_probability:
            return i
    return len(probabilities) - 1


def get_counts(probabilities: np.ndarray, shots: int = 1000) -> dict:
    histogram = {}
    for shot in range(shots):
        shot_index = measure_shot(probabilities=probabilities)
        key = str(shot_index)
        if key in histogram:
            histogram[key] = histogram[key] + 1
        else:
            histogram[key] = 1
    return histogram


def execute_circuit(num_qubits: int, operations: list) -> np.ndarray:
    dimension = 2 ** num_qubits
    current_state = np.zeros(dimension, dtype=complex)
    current_state[0] = 1

    hadamard = np.array([[1, 1], [1, -1]]) * 1 / np.sqrt(2)
    identity = np.array([[1, 0], [0, 1]])
    z_gate = np.array([[1, 0], [0, -1]])
    y_gate = np.array([[0, -1j], [1j, 0]])
    x_not = np.array([[0, 1], [1, 0]])

    p_0 = np.array([[1, 0], [0, 0]])
    p_1 = np.array([[0, 0], [0, 1]])

    for operation in operations:
        operation_result = None

        if operation["gate"] == "h":
            target_qubit = operation["target"]
            operations_list = []
            for qubit_index in range(num_qubits):
                if qubit_index == target_qubit:
                    operations_list.append(hadamard)
                else:
                    operations_list.append(identity)
            operation_result = _combine_operators(operations_list)

        elif operation["gate"] == "cx":
            parte_cero = []
            parte_uno = []
            target_qubit = operation["target"]
            control = operation["control"]
            for qubit_index in range(num_qubits):
                if qubit_index == control:
                    parte_cero.append(p_0)
                    parte_uno.append(p_1)
                elif qubit_index == target_qubit:
                    parte_cero.append(identity)
                    parte_uno.append(x_not)
                else:
                    parte_cero.append(identity)
                    parte_uno.append(identity)

            op_cero_final = _combine_operators(parte_cero)
            op_uno_final = _combine_operators(parte_uno)
            operation_result = op_cero_final + op_uno_final

        elif operation["gate"] == "x":
            target_qubit = operation["target"]
            operations_list = []
            for qubit_index in range(num_qubits):
                if qubit_index == target_qubit:
                    operations_list.append(x_not)
                else:
                    operations_list.append(identity)
            operation_result = _combine_operators(operations_list)

        elif operation["gate"] == "y":
            target_qubit = operation["target"]
            operations_list = []
            for qubit_index in range(num_qubits):
                if qubit_index == target_qubit:
                    operations_list.append(y_gate)
                else:
                    operations_list.append(identity)
            operation_result = _combine_operators(operations_list)

        elif operation["gate"] == "z":
            target_qubit = operation["target"]
            operations_list = []
            for qubit_index in range(num_qubits):
                if qubit_index == target_qubit:
                    operations_list.append(z_gate)
                else:
                    operations_list.append(identity)
            operation_result = _combine_operators(operations_list)

        if operation_result is not None:
            current_state = np.dot(operation_result, current_state)

    return current_state


if __name__ == '__main__':
    operations = [
        {'gate': 'h', 'target': 0},
        {'gate': 'cx', 'control': 0, 'target': 1}
    ]

    state = execute_circuit(num_qubits=2, operations=operations)

    probs = calculate_probabilities(current_state=state)

    histogram = get_counts(probabilities=probs, shots=1000)

    print(f"Estado Final: {state}")
    print(f"Probabilidades: {probs}")
    print(f"Histograma: {histogram}")
