import random

from Environment import Position, Environment
from SARSA import SARSA

EPISODES = 100000
EPSILON = 0.2
GAMMA = 1

initial_state = Position(0, 0)
sarsa = SARSA(epsilon=EPSILON, gamma=GAMMA)
env = Environment()

all_states = env.get_possible_start_states()

for i in range(EPISODES):
    current_state = random.choice(all_states)
    current_action = sarsa.pick_action(current_state)

    env.set_state(current_state)

    if i % 100 == 0:
        print(f"Run {i}/{EPISODES}")

    while True:
        reward, next_state = env.do_action(current_action)
        next_action = sarsa.pick_action(next_state)

        # Update Values
        sarsa.update(current_state, current_action, next_state, next_action, reward)
        current_state, current_action = next_state, next_action

        if env.is_terminal_state(current_state):
            break

import plotly.graph_objects as go
import plotly.express as px

directions = [[4]*9 for _ in range(9)]
directions[6][5] = 5
directions[8][8] = 5

# Pick the best direction for every cell
best_action_for_state = sarsa.get_best_actions()
for state, action in best_action_for_state.items():
    directions[state.y][state.x] = action.value

directions_text = [['']*9 for _ in range(9)]
for i in range(len(directions_text)):
    for j in range(len(directions_text[i])):
        dir_val = directions[i][j]
        text = ''
        match dir_val:
            case 0:
                text = '↑'
            case 1:
                text = '→'
            case 2:
                text = '↓'
            case 3:
                text = '←'
        directions_text[i][j] = text

#fig = go.Figure(data=go.Heatmap(
#                    z=directions), y)
x = y = list(range(9))
"""
for i in range(len(directions)):
    for j in range(len(directions[i])):
        if directions[i][j] <= 3:
            directions[i][j] = 0
        if directions[i][j] == 4:
            directions[i][j] = 1
        if directions[i][j] == 5:
            directions[i][j] = 2
"""
fig2 = px.imshow(directions, x=x, y=y, color_continuous_scale='Viridis', aspect="auto")


fig2.update_traces(text=directions_text, texttemplate="%{text}")
fig2.update_layout(coloraxis_showscale=False)

fig2.write_image("plots/viridis.png")
#print("Result")
#print(sarsa._Q)
