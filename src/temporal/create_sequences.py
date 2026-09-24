import numpy as np


def create_delta_sequences(
    data,
    sequence_length=12
):
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        x = data[i:i + sequence_length]

        current_state = data[i + sequence_length - 1]
        next_state = data[i + sequence_length]

        delta = next_state - current_state

        sequences.append(x)
        targets.append(delta)

    return (
        np.array(sequences),
        np.array(targets)
    )

