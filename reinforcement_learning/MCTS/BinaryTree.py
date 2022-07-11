import random
import Levenshtein
import math


class Node:
    def __init__(self, address: str):
        self.left: Node = None
        self.right: Node = None
        self.root: bool = False
        self.address = address
        self.value: float = 0
        self.t = 0  # Total discovered value
        self.n = 0  # Number of visits


class BinaryTree:
    _B = 5
    _TAU = 3

    def __init__(self):
        self._root: Node = Node(address='0')
        self._leaf_nodes: list[Node] = []

    def create_tree(self, depth: int) -> None:
        self._root: Node = Node(address='')
        self._root.root = True
        for i in range(depth):
            leaf_nodes = self._get_leaf_nodes()
            for node in leaf_nodes:
                current_node_address = node.address
                node.left = Node(address=current_node_address + 'L')
                node.right = Node(address=current_node_address + 'R')
        self._root.address = '0'

    def assign_values_to_leaf_nodes(self):
        leaf_nodes = self._get_leaf_nodes()

        # First, pick a random leaf-node as your target node and let’s denote its address as At
        target_node = random.choice(leaf_nodes)
        self.target_node = target_node  # For debugging
        print(f"Target node: {target_node.address}")

        # Assign value to every leaf node
        for node in leaf_nodes:
            distance = Levenshtein.distance(target_node.address, node.address)
            node_value = BinaryTree._B * math.e ** (-distance / BinaryTree._TAU)
            node.value = node_value

    def print_tree(self, node, level=0) -> None:
        if node is not None:
            self.print_tree(node.left, level + 1)
            print(' ' * 4 * level + '->', node.address)
            self.print_tree(node.right, level + 1)

    def print_leaf_nodes(self) -> None:
        self._leaf_nodes = []
        self._assign_leaf_nodes(node=self._root)
        print([f"{node.address}: {node.value}" for node in self._leaf_nodes])

    def _get_leaf_nodes(self) -> list[Node]:
        self._leaf_nodes = []
        self._assign_leaf_nodes(node=self._root)
        return self._leaf_nodes

    def _assign_leaf_nodes(self, node: Node) -> None:
        if not node.left and not node.right:
            self._leaf_nodes.append(node)
        if node.left:
            self._assign_leaf_nodes(node.left)
        if node.right:
            self._assign_leaf_nodes(node.right)

    @staticmethod
    def is_terminal_node(node: Node) -> bool:
        """Last node in tree"""
        return not bool(node.right)

    @property
    def root(self):
        return self._root

    @property
    def leaf_nodes(self):
        return self._leaf_nodes
