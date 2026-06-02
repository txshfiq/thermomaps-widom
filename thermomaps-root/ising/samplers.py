import numpy as np
import random
from collections import deque
from typing import Callable, Dict, List, Type
from abc import ABC, abstractmethod
from ising.base import IsingModel
from lattice import adjacency_list

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Sampler(ABC):
    """
    Abstract base class for samplers in the Ising model.

    A sampler is a method for updating the state of the system.

    Attributes:
        name (str): The name of the sampler.
    """

    def __init__(self, ising_model: Type[IsingModel]):
        """
        Initialize a Sampler.
        """
        self.IM = ising_model

    @abstractmethod
    def update(self):
        """
        Perform one update step.

        This is an abstract method that must be overridden by subclasses.
        """
        pass

class SwendsenWangSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SwendsenWang"

    def update(self):
        """
        Perform one update step using the Swendsen-Wang algorithm.
        """
        # Initialize the clusters from the current lattice configuration
        labels = self.initialize_clusters()

        # Build the clusters of connected spins
        self.build_clusters(labels)

        # Flip the clusters
        self.flip_clusters(labels)

        return self.IM.lattice

    def initialize_clusters(self) -> Dict[int, int]:
        """
        Initialize the clusters for the Swendsen-Wang algorithm.

        Returns:
            Dict[int, int]: A dictionary mapping each site to its cluster label.
        """
        return {i: i for i in range(self.IM.size * self.IM.size)}

    def find_root(self, site: int, labels: Dict[int, int]) -> int:
        """
        Find the root of the cluster that a site belongs to.

        Args:
            site (int): The linear index of the site.
            labels (Dict[int, int]): A dictionary mapping each site to its cluster label.

        Returns:
            int: The root of the cluster that the site belongs to.
        """
        while site != labels[site]:
            site = labels[site]
        return site

    def union(self, site1: int, site2: int, labels: Dict[int, int]):
        """
        Merge the clusters of two sites.

        Args:
            site1 (int): The linear index of the first site.
            site2 (int): The linear index of the second site.
            labels (Dict[int, int]): A dictionary mapping each site to its cluster label.
        """
        root1, root2 = self.find_root(site1, labels), self.find_root(site2, labels)
        if root1 != root2:
            labels[root2] = root1

    def build_clusters(self, labels: Dict[int, int]):
        """
        Build clusters of connected spins in the same state.
        """
        px = 1 - np.exp(-2 * self.IM.Jx / self.IM.temp)
        py = 1 - np.exp(-2 * self.IM.Jy / self.IM.temp)

        for x in range(self.IM.size):
            for y in range(self.IM.size):
                if x + 1 < self.IM.size and self.IM.lattice[x, y] == self.IM.lattice[x + 1, y] and random.random() < px:
                    self.union(x * self.IM.size + y, (x + 1) * self.IM.size + y, labels)
                if y + 1 < self.IM.size and self.IM.lattice[x, y] == self.IM.lattice[x, y + 1] and random.random() < py:
                    self.union(x * self.IM.size + y, x * self.IM.size + (y + 1), labels)

    def flip_clusters(self, labels: Dict[int, int]):
        """
        Flip clusters of connected spins.
        """
        should_flip = {root: random.choice([True, False]) for root in set(labels.values())}

        for x in range(self.IM.size):
            for y in range(self.IM.size):
                root = self.find_root(x * self.IM.size + y, labels)
                if should_flip[root]:
                    self.IM.lattice[x, y] *= -1

class SingleSpinFlipSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SingleSiteAndClusterFlip"

    def update(self):
        """
        Perform one update step using the Metropolis-Hastings algorithm.
        """
        nodes_size = len(self.IM.lattice)

        for _ in range(nodes_size):
            self.metropolis_hastings_step()

        return self.IM.lattice

    def cluster_finder(self, nodes, lattice_adjacency_list, start_node):
        """
        Find all nodes in the same cluster as the start node.

        Args:
            nodes (list): The list of nodes in the lattice.
            lattice_adjacency_list (list): The adjacency list of the lattice.
            start_node (int): The starting node.

        Returns:
            list: A list of all nodes in the same cluster.
        """
        visited = [False] * len(nodes)
        queue = deque([start_node])
        cluster = []
        target_value = nodes[start_node]

        visited[start_node] = True

        while queue:
            current = queue.popleft()
            cluster.append(current)
            for neighbor in lattice_adjacency_list[current]:
                if not visited[neighbor] and nodes[neighbor] == target_value:
                    visited[neighbor] = True
                    queue.append(neighbor)

        return cluster

    def metropolis_hastings_step(self):
        """
        Perform a single Monte-Carlo sampling step using the Metropolis-Hastings algorithm.
        """

        p = 0.9
        M = self.IM.M
        nodes = self.IM.lattice
        lattice_adjacency_list = adjacency_list(self.IM.size)

        A_remove_prob = min(1.0, (1.0 / (self.IM.z * M * p)))
        A_insert_prob = min(1.0, (self.IM.z * M * p))

        i = random.randint(0, len(nodes) - 1)   # Choose a site at random
        k = random.randint(1, M)                # Choose a color at random

        if nodes[i] != 0:

            if random.random() < p:

                if random.random() < A_remove_prob:
                    nodes[i] = 0
                else:
                    return

            else:
                cluster = self.cluster_finder(nodes, lattice_adjacency_list, i)

                col = random.randint(1, M)
                while col == nodes[i]:
                    col = random.randint(1, M)

                for v in cluster:
                    nodes[v] = col

        else:

            if random.random() < A_insert_prob:

                conflict = False

                for j in range(len(lattice_adjacency_list[i])):

                    index = lattice_adjacency_list[i][j]

                    if k != nodes[index] and nodes[index] != 0:
                        conflict = True
                        break

                if conflict is False:
                    nodes[i] = k
                else:
                    return

            else:
                return
