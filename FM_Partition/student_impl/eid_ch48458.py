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

    def compute_net_distribution(
        self, partition: List[int]
    ) -> List[List[int]]:
        net_distribution = [[0, 0] for i in range(self.n_nets)]
        for node, nets in enumerate(self.node2net_map):
            part = partition[node]  # which partition, 0 or 1
            for net in nets:
                net_distribution[net][part] += 1
        return net_distribution

    def gain2index(self, gain: int) -> int:
        return gain + self.max_node_degree

    def build_bucket_lists(self):
        # Compute max node degree
        self.compute_max_node_degree()
        print('[Info] Max node degree:', self.max_node_degree)
        
        # Compute bucket size = 2 * max_node_degree + 1
        self.bucket_size = 2 * self.max_node_degree + 1

        # Build bucket lists
        self.bucket_lists = [
            [[] for _ in range(self.bucket_size)], 
            [[] for _ in range(self.bucket_size)] 
        ]

        # Build net partition (num_nodes_on_left, num_nodes_on_right)
        self.net_distribution = self.compute_net_distribution(
            self.initial_partition
        )

        self.max_gains = [-self.max_node_degree, -self.max_node_degree]
        self.node_gains = []
        for node, nets in enumerate(self.node2net_map):
            gain = 0
            part = self.initial_partition[node]
            opposite_part = (1 if part == 0 else 0)
            for net in nets:
                if len(self.net2node_map[net]) > 1:
                    if self.net_distribution[net][part] == 1:
                        gain += 1
                    elif self.net_distribution[net][opposite_part] == 0:
                        gain -= 1

            # Insert node to the bucket list according to its gain
            index = self.gain2index(gain)
            self.bucket_lists[part][index].append(node)
            self.node_gains.append(gain)

            if self.max_gains[part] < gain:
                self.max_gains[part] = gain

        # print(self.bucket_lists[0])
        # print(self.bucket_lists[1])
        # print(self.max_gains)
        # print(self.node_gains)

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
        initial_partition_node = self.convert_partition2tuple_node(
            self.initial_partition
        )
        initial_cutsize = self.compute_cut_size(initial_partition_node)
        # print('Net2node', self.net2node_map)
        # print('Node2net', self.node2net_map)
        # print('Initial partition', initial_partition_node)
        self.cut_size = initial_cutsize
        self.cut_size_list = [initial_cutsize]
        self.best_cut_size = initial_cutsize
        self.best_step = 0
        self.swap_node_list = []
        self.partition_sizes = [
            len(initial_partition_node[0]), len(initial_partition_node[1])
        ]
        print(f'[Info] Partition sizes: {self.partition_sizes}')
        print(f'[Info] Initial cutsize: {initial_cutsize}')

        # Build bucket lists
        self.build_bucket_lists()
        self.locked = [False for _ in range(self.n_nodes)]

    def get_valid_moves(self) -> Tuple[bool, bool]:
        valid_move_0 = (
            (
                min(self.partition_sizes[0] - 1, self.partition_sizes[1] + 1)
                / self.n_nodes
            )
            >= (self.min_cut_ratio - self.min_cut_ratio_epsilon)
        )
        valid_move_1 = (
            (
                min(self.partition_sizes[0] + 1, self.partition_sizes[1] - 1)
                / self.n_nodes
            )
            >= (self.min_cut_ratio - self.min_cut_ratio_epsilon)
        )
        return valid_move_0, valid_move_1

    def get_max_gain_and_node(self) -> Tuple[int, int]:
        """
            Return:
                Tuple[int, int]: max_gain, node
        """
        max_gain = -self.max_node_degree
        node = 0
        
        valid_moves = self.get_valid_moves()
        max_gains = self.max_gains[0], self.max_gains[1]
        indexes = (
            self.gain2index(self.max_gains[0]),
            self.gain2index(self.max_gains[1])
        )
        if valid_moves[0] == False:
            max_gain = max_gains[1]
            assert len(self.bucket_lists[1][indexes[1]]) > 0
            node = min(self.bucket_lists[1][indexes[1]])
        elif valid_moves[1] == False:
            max_gain = max_gains[0]
            assert len(self.bucket_lists[0][indexes[0]]) > 0
            node = min(self.bucket_lists[0][indexes[0]])
        elif max_gains[0] < max_gains[1]:
            max_gain = max_gains[1]
            assert len(self.bucket_lists[1][indexes[1]]) > 0
            node = min(self.bucket_lists[1][indexes[1]])
        elif max_gains[0] > max_gains[1]:
            max_gain = max_gains[0]
            assert len(self.bucket_lists[0][indexes[0]]) > 0
            node = min(self.bucket_lists[0][indexes[0]])
        else:
            assert valid_moves == (True, True)
            assert max_gains[0] == max_gains[1]
            assert len(self.bucket_lists[0][indexes[0]]) > 0
            assert len(self.bucket_lists[1][indexes[1]]) > 0
            max_gain = max_gains[0]
            node = min(self.bucket_lists[0][indexes[0]] + 
                       self.bucket_lists[1][indexes[1]])
        return max_gain, node

    def remove_node_and_update_max_gain(self, node, part, gain):
        index = self.gain2index(gain)
        self.bucket_lists[part][index].remove(node)
        if (self.max_gains[part] == gain and 
                len(self.bucket_lists[part][index]) == 0
            ):
            index -= 1
            while index >= -1:
                if len(self.bucket_lists[part][index]) > 0:
                    break
                index -= 1
            self.max_gains[part] = index - self.max_node_degree
    
    def insert_node_and_update_max_gain(self, node, part, gain):
        index = self.gain2index(gain)
        self.bucket_lists[part][index].append(node)
        if gain > self.max_gains[part]:
            self.max_gains[part] = gain

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
        for step in range(1, self.n_nodes + 1):

            if self.get_valid_moves() == (False, False):
                break

            max_gain, node = self.get_max_gain_and_node()
            
            # Add a swap node
            self.swap_node_list.append(node)

            # Update cutsize
            self.cut_size -= max_gain
            self.cut_size_list.append(self.cut_size)
            if self.cut_size < self.best_cut_size:
                self.best_cut_size = self.cut_size
                self.best_step = step
            
            # print(f'[Info] Step {step}: Gain {max_gain}, Node {node}, '
                  # f'Cutsize {self.cut_size}')
            
            # Locked and move
            assert self.locked[node] == False
            self.locked[node] = True

            # Move node to the opposite partition
            part = self.initial_partition[node]
            opposite_part = (1 if part == 0 else 0)
            self.partition_sizes[part] -= 1
            self.partition_sizes[opposite_part] += 1
            
            # Update net distribution
            for net in self.node2net_map[node]:
                self.net_distribution[net][part] -= 1
                self.net_distribution[net][opposite_part] += 1

            # Update bucket lists
            ### Remove current node
            self.remove_node_and_update_max_gain(node, part, max_gain)

            ### Update unlocked node connected to current node
            for net in self.node2net_map[node]:
                for net_node in self.net2node_map[net]:
                    if self.locked[net_node] == True:
                        continue
                    gain = 0
                    net_node_part = self.initial_partition[net_node]
                    net_node_oppo_part = (1 if net_node_part == 0 else 0)
                    for net_node_net in self.node2net_map[net_node]:
                        if len(self.net2node_map[net_node_net]) > 1:
                            # print('  ', self.net_distribution[net_node_net])
                            if self.net_distribution[net_node_net][net_node_part] == 1:
                                gain += 1
                            elif self.net_distribution[net_node_net][net_node_oppo_part] == 0:
                                gain -= 1
                    # print(f'Node {net_node} gain {gain}')
                    
                    # Update
                    if gain != self.node_gains[net_node]:
                        self.remove_node_and_update_max_gain(
                            net_node, net_node_part, self.node_gains[net_node]
                        )
                        self.insert_node_and_update_max_gain(
                            net_node, net_node_part, gain
                        )
                        self.node_gains[net_node] = gain

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
