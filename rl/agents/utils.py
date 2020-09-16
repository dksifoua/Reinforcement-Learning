import numpy as np


def argmax(q_values: np.ndarray) -> int:
    """
    Return the argmax of q_values. If there's ties, we randomly select one of them.

    Numpy's argmax function returns the  first instance of the highest value. We do not want that to happen as it
    biases the agent to choose a specific action in the case of ties. Instead we want to break ties between the highest
    values randomly. So we are going to implement our own argmax function. You may want to look at np.random.choice to
    randomly select from a list of values.

    Args:
        q_values: np.array
            The action values

    Returns: int
        The index representing the action to be taken
    """
    max_q = float('-inf')
    ties = []
    for index, q_value in enumerate(q_values):
        if q_value >= max_q:
            if q_value == max_q:
                ties.append(index)
            else:
                ties = [index]
            max_q = q_value
    return ties[np.random.choice(len(ties))]
