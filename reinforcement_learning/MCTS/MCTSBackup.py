import random
import numpy as np

from BinaryTree import BinaryTree, Node


class MCTS:

    def __init__(self, computational_budget: int = 20, exploration_weight: int = 2):
        self._computational_budget = computational_budget
        self._exploration_weight = exploration_weight  # c

    def find_optimal_node(self, binary_tree: BinaryTree) -> Node:
        node = binary_tree.root
        while True:
            if BinaryTree.is_leaf_node(node=node):
                break
            for _ in range(self._computational_budget):
                self._expansion(node)
            node = self._select_child_node(node)
        return node

    def _expansion(self, node: Node) -> float:
        """Expansion with backtrack"""
        node.n += 1
        if BinaryTree.is_leaf_node(node=node):
            # Backtrack
            return node.value
        child_node = self._select_child_node(node=node)
        value = self._expansion(node=child_node)
        node.t += value
        return value

    def _select_child_node(self, node: Node) -> Node:
        """Select child node based on Upper Confidence Bound (UCB)"""
        if node.left.n == 0: return node.left
        if node.right.n == 0: return node.right

        if self._upper_confidence_bound(node.left, node.n) > self._upper_confidence_bound(node.right, node.n):
            return node.left
        return node.right

    def _upper_confidence_bound(self, node: Node, nbr_parent_visits: int) -> float:
        """Calculate UCB value of node"""
        mean = node.t / node.n
        return mean + self._exploration_weight * np.sqrt(np.log(nbr_parent_visits) / node.n)

    @staticmethod
    def _roll_out(node: Node) -> float:
        """ Random roll-out starting from given root node"""
        visited_nodes = [node]
        node.n = 1
        while node.right:
            node = random.choice([node.right, node.left])
            node.n = 1
            visited_nodes.append(node)
        for v_node in visited_nodes:
            v_node.t = node.value
        return node.value
