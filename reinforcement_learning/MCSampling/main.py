# Monte Carlo Sampling
# Approaches:
import plotly.graph_objects as go
import numpy as np
import math

from Strategies import PercentRule, HigherThanNumberRule

NUMBER_OF_SAMPLES = 1000

N = 10000

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

fig = go.Figure()

NUMBER_OF_SAMPLES = 1000

APARTMENTS_LENGTH = [10, 100, 1000]

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

results = [[], [], [], [], [], []]

for N in APARTMENTS_LENGTH:
    for idx, strategy in enumerate(strategies):
        correct_answers = 0
        for _ in range(NUMBER_OF_SAMPLES):
            apartments = np.random.rand(N)
            highest_apartment = np.max(apartments)
            picked_apartment = strategy.pick_apartment(apartments)
            if picked_apartment == highest_apartment:
                correct_answers += 1
        result = np.round(correct_answers/NUMBER_OF_SAMPLES, 2)
        results[idx].append(result)

fig = go.Figure()

colors = [
    '#92BDA3',
    '#A1BA89',
    '#A5CC6B',
    '#153B50',
    '#074363',
    '#283740',   
]
    
for idx, result in enumerate(results):
    fig.add_trace(go.Bar(
    x=list(map(str, APARTMENTS_LENGTH)),
    y=result,
    name=strategies[idx].text,
    marker_color=colors[idx]
))
    
fig.update_xaxes(title_text='Size N')
fig.update_yaxes(title_text='Success in %')
    

    
fig.update_layout(barmode='group')
fig.write_image('plots/plot.png', scale=3)
fig.show()
