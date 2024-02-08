#######################################################################
# Implementation of FM partition
# You need to implement initialize() and partition_one_pass()
# All codes should be inside FM_Partition class
# Name: Chen-Hao Hsu
# UT EID: ch48458
#######################################################################

from typing import List, Tuple

import numpy as np

from .p1_partition_base import FM_Partition_Base

__all__ = ["FM_Partition"]

class FM_Partition(FM_Partition_Base):
    def __init__(self) -> None:
        super().__init__()

    class DoublyLinkedNode():
        def __init__(self) -> None:
            self.prev = None
            self.next = None
        
        def __repr__(self):
            return f'({self.prev}, {self.next})'
        
        def __str__(self):
            return f'({self.prev}, {self.next})'

    class DoublyLinkedList():
        def __init__(self) -> None:
            self.first_node = None
            self.last_node = None
        
        def empty(self) -> bool:
            return self.first_node == None
        
        def show(self, ln):
            node = self.first_node
            while node != None:
                print(f' {node}')
                node = ln[node].next

    def get_initial_partition(self) -> List[int]:
        """ Get an initial partition
        """
        initial_partition = (
            [0 for _ in range(self.n_nodes // 2) ] +
            [1 for _ in range(self.n_nodes - (self.n_nodes // 2))]
        )
        return initial_partition

    def convert_partition2tuple_node(
        self, partition: List[int]
    ) -> Tuple[List[int], List[int]]:
        partition_node = ([], [])
        for i, p in enumerate(partition):
            partition_node[p].append(i)
        return partition_node
    
    def convert_partition2tuple_node_name(
        self, partition: List[int]
    ) -> Tuple[List[str], List[str]]:
        partition_node_name = ([], [])
        for i, p in enumerate(partition):
            partition_node_name[p].append(self.node2node_name_map[i])
        return partition_node_name
    
    def compute_max_node_degree(self):
        """ Compute maximum node degree for max/min gains
        """
        node_degree_list = [
            len(self.node2net_map[i]) for i in range(self.n_nodes)
        ]
        self.max_node_degree = max(node_degree_list)

    def compute_net_partition(
        self, partition: List[int]
    ) -> List[List[int]]:
        net_partition = [[0, 0] for i in range(self.n_nets)]
        for node, nets in enumerate(self.node2net_map):
            part = partition[node]  # which partition, 0 or 1
            for net in nets:
                net_partition[net][part] += 1
        return net_partition

    def build_bucket_lists(self):
        # Compute max node degree
        self.compute_max_node_degree()
        print('[Info] Max node degree:', self.max_node_degree)
        
        # Compute bucket size = 2 * max_node_degree + 1
        self.bucket_size = 2 * self.max_node_degree + 1
        
        # List nodes
        # Nodes: n_0, n_1, ..., n_{n-1}
        # bucket list[0]: -P_min, 0, -P_max
        # bucket list[1]: -P_min, 0, -P_max
        self.ln = [ 
            self.DoublyLinkedNode() for _ in range(
                self.n_nodes + (2 * self.bucket_size)
            )
        ]

        # Build bucket lists
        self.bucket_lists = [
            [self.DoublyLinkedList() for _ in range(self.bucket_size)], 
            [self.DoublyLinkedList() for _ in range(self.bucket_size)] 
        ]

        # Build net partition (num_nodes_on_left, num_nodes_on_right)
        self.net_partition = self.compute_net_partition(self.initial_partition)

        for node, nets in enumerate(self.node2net_map):
            gain = 0
            part = self.initial_partition[node]
            opposite_part = (1 if part == 0 else 0)
            for net in nets:
                if (self.net_partition[net][part] == 1 and
                    self.net_partition[net][opposite_part] > 0
                ):
                    gain += 1
                elif (self.net_partition[net][opposite_part] == 0 and
                     self.net_partition[net][part] > 1
                ):
                    gain -= 1
            print(f'Node {node} gain {gain}')
        
        print(self.net_partition)
        print(self.bucket_lists)
        print(self.ln)

    # def insert_front(
            # self, node: int, part: int, gain: int
    # ):
        # """ Insert a node to the front
        # """
        # index = gain + self.max_node_degree  # bucket index
        # if self.bucket_lists[part][index].last_node == None:
            
            



    # def remove_node(self, node):


    def get_best_partition(self) -> List[int]:
        best_partition = self.initial_partition
        for i in range(self.best_step):
            node = self.swap_node_list[i]
            if best_partition[node] == 0:
                best_partition[node] = 1
            elif best_partition[node] == 1:
                best_partition[node] = 0
            else:
                print('[Error] Bad partition: node', node)
        return best_partition

    def initialize(self):
        """
            Initialize necessary data structures before starting
            solving the problem
        """
        print(f'[Info] #Nodes: {self.n_nodes}')
        print(f'[Info] #Nets: {self.n_nets}')
        
        self.cut_size = 0  # current cutsize of self.sol
        self.cut_size_list = []
        self.swap_node_list = []
        
        self.best_cut_size = 0  # best custsize of self.best_sol
        self.best_step = 0  # best step
        
        # Build node2net_map
        self.node2net_map = [[] for _ in range(self.n_nodes)]
        for net, nodes in enumerate(self.net2node_map):
            for node in nodes:
                self.node2net_map[node].append(net)
 
        # TODO initial solutions: block 0 and block 1
        # To ensure a deterministic solution,
        # use the following partition as the initial solution
        # sort the node names in alphabetical order
        # the first floor(N/2) nodes are in the first partition,
        # the rest N-floor(N/2) nodes are in the second partition
        # a_0, a_1, ..., a_N/2-1 | a_N/2, a_N/2+1, ..., a_N-1, if N even
        # a_0, a_1, ..., a_(N-3)/2 | a_(N-1)/2, ..., a_N-1, if N odd
        # ...
        # Example (even): 0 1 | 2 3 (n = 4)
        # Example (odd): 0 1 | 2 3 4 (n = 5)

        # TODO initialize any auxiliary data structure you need
        # e.g., node2net_map, cell gains, locked cells, etc.
        
        # Initial partition
        self.initial_partition = self.get_initial_partition()
        initial_cutsize = self.compute_cut_size(
            self.convert_partition2tuple_node(self.initial_partition)
        )
        print('Net2node', self.net2node_map)
        print('Node2net', self.node2net_map)
        print(self.convert_partition2tuple_node(self.initial_partition))
        self.cut_size = initial_cutsize
        self.best_cut_size = initial_cutsize
        self.cut_size_list.append(initial_cutsize)
        print('[Info] Initial cutsize: {initial_cutsize}')

        # Build bucket lists
        self.build_bucket_lists()

    def partition_one_pass(
        self
    ) -> Tuple[List[int], Tuple[List[str], List[str]], int]:
        """FM graph partition algorithm for one pas

        Return:
            cut_size_list (List[int]):
                contains the initial cut size and the cut size after each move
            best_sol (Tuple[List[str], List[str]]):
                The best partition solution is a tuple of two blocks.
                Each block is a list of node names.
                (Please use the original node names from the benchmark file.
                Hint: you might need to use node2node_name_map).
                If multiple solutions have the best cut size, return
                the first one.
            best_cut_size (int):
                The cut size corresponding to the best partition solution
        """
        # TODO implement your FM partition algorithm for one pass.
        # To make this method clean, you can extract subroutines
        # as methods of this class
        # But do not override methods in the parent class
        # Please strictly follow the return type requirement.

        # Get the best
        best_partition = self.get_best_partition()
        best_sol_node = self.convert_partition2tuple_node(best_partition)
        best_cut_size = self.compute_cut_size(best_sol_node)
        if best_cut_size != self.best_cut_size:
            print('[Warn] Inconsistent best cut sizes',
                  best_cut_size, self.best_cut_size)
            self.best_cut_size = best_cut_size
        best_sol = self.convert_partition2tuple_node_name(best_partition)

        return self.cut_size_list, best_sol, self.best_cut_size
