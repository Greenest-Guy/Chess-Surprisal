import numpy as np


def softmax_probability_all(z: list, temperature=1) -> dict:
    z = np.array(z, dtype=float)

    numerator = np.exp(z/temperature)
    denominator = np.sum(np.exp(z/temperature))
    prob = numerator / denominator

    return prob


if __name__ == '__main__':
    probs = softmax_probability_all([250, 200, 30, -200], temperature=50)

    for i in probs:
        print(f"{i*100:.4f}%")
