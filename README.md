# Chess Surprisal
Code utilizing information theory to calculate the surprisal (self-information) of a chess game, derived from Stockfish evaluations.

Note that this repository is intended to serve as demonstration code for an independent research paper: "*Quantifying the Degradation of Human Decision-Making Under Time Pressure Through the Self-Information of Chess*".

## Probability Framework

<p align="center">
$\displaystyle p_i=\frac{e^{E_i/\Delta_0}}{\sum_{x\in L}e^{E_x/\Delta_0}}$
</p>

```Python
def softmax_probability_all(evals: list, Ei: int, delta0: int):
        evals = np.array(evals, dtype=float)

        # Temperature Scaling
        scaled_evals = evals / delta0
        scaled_Ei = Ei / delta0

        # prevents exponential overflow
        max_val = np.max(scaled_evals)

        numerator = np.exp(scaled_Ei - max_val)
        denominator = np.sum(np.exp(scaled_evals - max_val))
        return numerator / denominator
```

## Surprisal Framework
<p align="center">
$C = \sum_{i=1}^{n}  -\log_2(p_i)$
</p>

```Python
def calculate_total_information_cost(probabilities: list):
        probabilities_np = np.array(probabilities, dtype=float)

        # Prevent log divide by zero
        probabilities_np = np.clip(probabilities_np, 1e-15, 1.0)

        total_surprisal = -np.sum(np.log2(probabilities_np))
        return total_surprisal
```
