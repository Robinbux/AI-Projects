from Environment import Environment, Position
from MonteCarloPolicyEvaluation import MonteCarloPolicyEvaluation
from Policies import EquiprobablePolicy

NBR_SAMPLES = 50

policy_eval = MonteCarloPolicyEvaluation(NBR_SAMPLES)
policy = EquiprobablePolicy()

map = policy_eval.evaluate(policy)