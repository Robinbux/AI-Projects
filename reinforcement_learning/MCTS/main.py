import numpy as np

from BinaryTree import BinaryTree
from MCTS import MCTS

binary_tree = BinaryTree()
binary_tree.create_tree(12)

binary_tree.assign_values_to_leaf_nodes()

start = 0
stop = 10.5
step = 0.5

weights = np.round(np.arange(start, stop, step), 2)
budgets = np.round(np.arange(10, 110, 10), 2)

NBR_RUNS = 50
COMPUTATIONAL_BUDGET = 100

total_y = []
total_x = []

results = {}

for budget in budgets:
    y = []
    x = []
    for weight in weights:
        mcts = MCTS(computational_budget=budget, exploration_weight=weight)
        correct_results = 0
        for run in range(NBR_RUNS):
            optimal_node = mcts.find_optimal_node(binary_tree=binary_tree)
            if optimal_node.address == binary_tree.target_node.address:
                correct_results += 1
        print("--------------------")
        print(f"Budget: {budget}")
        print(f"Weight: {weight}")
        print(f"{correct_results}/{NBR_RUNS} ≈ {correct_results/NBR_RUNS}% correct")
        x.append(weight)
        y.append(correct_results/NBR_RUNS)
    results[str(budget)] = (x, y)
    print("Current result:")
    print(results)

print(results)

#optimal_node = mcts.find_optimal_node(binary_tree=binary_tree)

#print(f"\nFound Optimal Node: {optimal_node.address}, Value: {optimal_node.value}")
#print(f"Real Optimal Node: {binary_tree.target_node.address}, Value: {binary_tree.target_node.value}")