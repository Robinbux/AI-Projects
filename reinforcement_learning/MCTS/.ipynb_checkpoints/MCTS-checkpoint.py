import random
import numpy as np
import copy

from BinaryTree import BinaryTree, Node


class MCTS:

    def __init__(self, computational_budget: int = 20, exploration_weight: int = 2):
        self._computational_budget = computational_budget
        self._exploration_weight = exploration_weight  # c

    def find_optimal_node(self, binary_tree: BinaryTree) -> Node:
        binary_tree = copy.deepcopy(binary_tree)
        current_root = binary_tree.root
        while True:
            node_path = []
            root_copy = copy.deepcopy(current_root)
            current_node = current_root
            for _ in range(self._computational_budget):
                while True:
                    node_path.append(current_node)
                    if BinaryTree.is_terminal_node(current_node):
                        for node in node_path:
                            node.t += current_node.value
                            node.n += 1
                        node_path = []
                        current_node = current_root
                        break
                    if self._is_leaf_node(current_node):
                        if current_node.n == 0:
                            # Do rollout
                            rollout_value = self._roll_out(current_node)
                            for node in node_path:
                                node.t += rollout_value
                                node.n += 1
                            node_path = []
                            current_node = current_root
                            break
                        else:
                            current_node = current_node.left
                    else:
                        current_node = self._select_child_node(current_node)
            # Chose child node and start all over
            left_avg = current_root.left.t / current_root.left.n
            right_avg = current_root.right.t / current_root.right.n
            current_root = root_copy.left if left_avg > right_avg else root_copy.right
            if BinaryTree.is_terminal_node(current_root):
                return current_root

    # @staticmethod
    # def _forget_old_values(node: Node) -> None:
    #     if node:
    #         MCTS._forget_old_values(node.left)
    #
    #         # now recur on right child
    #         MCTS._forget_old_values(node.right)

    @staticmethod
    def _roll_out(node: Node) -> float:
        """ Random roll-out starting from given root node"""
        while not BinaryTree.is_terminal_node(node):
            node = random.choice([node.right, node.left])
        return node.value

    @staticmethod
    def _is_leaf_node(node: Node) -> bool:
        """A node in this term is a leaf node, if both childs have n = 0"""
        return node.left.n == 0 and node.right.n == 0 and not node.root

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

