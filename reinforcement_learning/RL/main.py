from Environment import Environment, Position
from MonteCarloPolicyEvaluation import MonteCarloPolicyEvaluation
from Policies import EquiprobablePolicy
import plotly.express as px


NBR_SAMPLES = 5000

policy_eval = MonteCarloPolicyEvaluation(NBR_SAMPLES)
policy = EquiprobablePolicy()

map = policy_eval.evaluate(policy)

fig = px.imshow(map, text_auto=True)
fig.show()