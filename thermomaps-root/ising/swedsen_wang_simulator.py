from collections import deque

import numpy as np
import random
from typing import Callable, Dict, List
from abc import ABC, abstractmethod
from lattice import adjacency_list

class Observable(ABC):
    """
    Abstract base class for observables in the Ising model.

    An observable is a physical quantity that can be measured in a simulation.

    Attributes:
        name (str): The name of the observable.
    """

    def __init__(self, name: str):
        """
        Initialize an Observable.

        Args:
            name (str): The name of the observable.
        """
        # Name of the observable
        self.name = name

    @abstractmethod
    def evaluate(self, lattice: np.ndarray) -> float:
        """
        Calculate the value of the observable for a given lattice.

        This is an abstract method that must be overridden by subclasses.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The value of the observable.
        """
        pass

class Demixed(Observable):
    """
    Class for calculating the demixed parameter of a lattice.

    The demixed parameter is defined as the absolute value of the sum of the complex phases of the spins.
    This class inherits from the Observable class.

    Attributes:
        name (str): The name of the observable, "demixed".
    """

    def __init__(self):
        """
        Initialize an Demixed observable.

        """
        super().__init__("demixed")

    def evaluate(self, lattice: np.ndarray) -> float:
        """
        Calculate the demixed parameter of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The demixed parameter of the lattice.
        """
        frame = lattice
        M = 2

        density = sum(n != 0 for n in frame) / len(frame)

        total = 0 + 0j

        for i in range(1, M + 1):
            angle = 2 * np.pi * (i - 1) / M
            euler = np.exp(-1j * angle)

            N_i = frame.count(i)

            m_i = N_i / (density * len(frame))

            total += m_i * euler

        return abs(total)
        

class Magnetization(Observable):
    """
    Class for calculating the magnetization of a lattice.

    Magnetization is defined as the mean value of the spins in the lattice.
    This class inherits from the Observable class.

    Attributes:
        name (str): The name of the observable, "magnetization".
    """

    def __init__(self):
        """
        Initialize a Magnetization observable.
        """
        super().__init__("magnetization")

    def evaluate(self, lattice: np.ndarray) -> float:
        """
        Calculate the magnetization of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The magnetization of the lattice.
        """
        # Magnetization is the mean value of the spins
        return np.mean(lattice)


class IsingSwendsenWang:
    """
    Class for simulating the Ising model using the Swendsen-Wang algorithm.

    Attributes:
        size (int): The size of the lattice.
        lattice (np.ndarray): The lattice of spins.
        warm_up (int): The number of warm-up steps.
        temp (float): The temperature of the system.
        snapshots (list): A list to store snapshots of the system state.
    """
    
    def lattice_init(self, z: float, size: int, M: int = 2, lattice_type: str = "square"):
        """
        Initialize the lattice for the Ising model.

        Args:
            z (float): The activity of the system.
            size (int): The size of the lattice.
            M (int, optional): The number of species. Defaults to 2.
        """
        lattice_adjacency_list = adjacency_list(size, lattice_type)
        nodes = [0] * (size * size)
        
        p = (M * z) / ((M * z) + 1)


        for i in range(len(nodes)):

            success = random.random() < p

            if not success:
                continue

            k = random.randint(1, M)

            conflict = False

            # Check neighbors
            for j in range(len(lattice_adjacency_list[i])):
                index = lattice_adjacency_list[i][j]

                if k != nodes[index] and nodes[index] != 0:
                    conflict = True
                    break

            # Update node
            if not conflict:
                nodes[i] = k
            else:
                nodes[i] = 0

        return nodes


    def __init__(self, size: int, warm_up: int, z: float, M: int, lattice_type: str = "square"):
        """
        Initialize an IsingSwendsenWang simulator.

        Args:
            size (int): The size of the lattice.
            warm_up (int): The number of warm-up steps.
            z (float): The activity of the system.
            M (int): The number of species.
            lattice_type (str, optional): The type of lattice. Defaults to "square".
        """
        # Size of the lattice
        self.size = size

        # Activity of the system
        self.z = z

        # Number of species
        self.M = M

        # Type of lattice
        self.lattice_type = lattice_type

        # Initialize the lattice with random spins
        self.lattice = self.lattice_init(z, size, M, lattice_type)

        # Number of warm-up steps
        self.warm_up = warm_up

        # List to store snapshots of the system state
        self.snapshots = []

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

    def swendsen_wang_step(self):
        
        p = 0.9
        M = self.M
        nodes = self.lattice
        lattice_adjacency_list = adjacency_list(self.size)

        A_remove_prob = min(1.0, (1.0 / (self.z * M * p)))
        A_insert_prob = min(1.0, (self.z * M * p))

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

    def simulate(self, steps: int, observables: List[Callable[[np.ndarray], float]], sampling_frequency: int):
        """
        Simulate the Ising model using the Swendsen-Wang algorithm.

        This function performs a number of steps of the Swendsen-Wang algorithm, and periodically
        samples the state of the system and the values of the observables.

        Args:
            steps (int): The number of steps to perform.
            observables (List[Callable[[np.ndarray], float]]): A list of observables to measure.
            sampling_frequency (int): The frequency at which to sample the state of the system and the observables.

        Returns:
            dict: A dictionary containing the sampled states of the system and the values of the observables.
        """
        # Perform the warm-up steps
        for _ in range(self.warm_up):
            self.swendsen_wang_step()

        nodes_size = len(self.lattice)

        # Perform the simulation steps
        for i in range(steps):
            for _ in range(nodes_size):
                self.swendsen_wang_step()

                # Sample the state of the system and the observables
                if i % sampling_frequency == 0:
                    if not self.snapshots:
                        # Initialize the dictionary with lists
                        self.snapshots = {'lattice': [self.lattice.copy()]}
                        for obs in observables:
                            self.snapshots[obs.name] = [obs.evaluate(self.lattice)]
                    else:
                        # Append to the existing lists
                        self.snapshots['lattice'].append(self.lattice.copy())
                        for obs in observables:
                            self.snapshots[obs.name].append(obs.evaluate(self.lattice))

        # Convert lists to numpy arrays
        for key in self.snapshots:
            self.snapshots[key] = np.array(self.snapshots[key])

        # Return the sampled states and observables
        return self.snapshots

    def save_snapshots(self, filename: str, metadata: dict = None):
        """
        Save the snapshots of the system state and the observables to a file.

        This function saves the snapshots of the system state and the observables to a compressed
        numpy (.npz) file. Additional metadata can also be included in the file.

        Args:
            filename (str): The name of the file to save the snapshots to.
            metadata (dict, optional): A dictionary of metadata to include in the file.
        """
        # If metadata is provided, include it in the file
        if metadata:
            np.savez_compressed(f"snapshots/{filename}.npz", **self.snapshots, **metadata)
        else:
            # Otherwise, just save the snapshots
            np.savez_compressed(f"snapshots/{filename}.npz", **self.snapshots)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--size", type=int, default=20)
    argparser.add_argument("--warm-up", type=int, default=10000)
    argparser.add_argument("--steps", type=int, default=100000)
    argparser.add_argument("--z", type=float, default=0.1)
    argparser.add_argument("--sampling-frequency", type=int, default=1)
    argparser.add_argument("--filename", type=str, default="ising")
    argparser.add_argument("--M", type=int, default=2)
    args = argparser.parse_args()

    ising = IsingSwendsenWang(args.size, args.warm_up, args.z, args.M)
    results = ising.simulate(args.steps, [Demixed(), Magnetization()], args.sampling_frequency)

    ising.save_snapshots(args.filename)