# Monte Carlo Sampling
# Approaches:

import numpy as np
import math

from Strategies import PercentRule, HigherThanNumberRule

NUMBER_OF_SAMPLES = 1000

N = 1000
SPLIT_POINT = int(1 / math.e * N)

strategies = [
    PercentRule(
        percentage=0.3,
        text="30% Split"
    ),
    PercentRule(
        percentage=1 / math.e,
        text="1/e Split"
    ),
    PercentRule(
        percentage=0.5,
        text="50% Split"
    ),
    HigherThanNumberRule(
        number=0.95,
        text="Number 0.95"
    ),
    HigherThanNumberRule(
        number=0.99,
        text="Number 0.99"
    ),
    HigherThanNumberRule(
        number=0.999,
        text="Number 0.999"
    )
]


correct_answers = 0
for _ in range(NUMBER_OF_SAMPLES):
    apartments = np.random.rand(N)
    ap_to_split = apartments[:SPLIT_POINT]
    ap_from_split = apartments[SPLIT_POINT:]
    total_highest_ap = np.max(apartments)

    highest_ap_until_split_point = np.max(ap_to_split)
    next_ap_that_is_higher = ap_from_split[np.argmax(ap_from_split > highest_ap_until_split_point)]

    if next_ap_that_is_higher == total_highest_ap:
        correct_answers += 1

print(f"{correct_answers}/{N} ≈ {correct_answers / N * 100}% correct answers.")
