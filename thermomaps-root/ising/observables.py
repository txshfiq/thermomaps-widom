import numpy as np
from abc import ABC, abstractmethod
from data.observables import Observable

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Demixed(Observable):
    """
    Class for calculating the demixed parameter of a lattice.

    The demixed parameter is a measure of the degree to which there is a separation of different species in the lattice.
    This class inherits from the Observable class.

    Attributes:
        name (str): The name of the observable, "demixed".
        Jx (float): The interaction energy along the x direction.
        Jy (float): The interaction energy along the y direction.
    """

    def __init__(self, name: str = 'demixed'):
        """
        Initialize a Demixed observable.
        """
        super().__init__("demixed")


    def evaluate_frame(self, lattice: np.ndarray) -> float:
        """
        Calculate the demixed parameter of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The demixed parameter of the lattice.
        """
        M = 2
        frame = lattice.tolist()

        density = sum(n != 0 for n in frame) / len(frame)

        total = 0 + 0j

        for i in range(1, M + 1):
            angle = 2 * np.pi * (i - 1) / M
            euler = np.exp(-1j * angle)

            N_i = frame.count(i)

            m_i = N_i / (density * len(frame))

            total += m_i * euler

        return abs(total)
        




    def evaluate(self, time_series: np.ndarray) -> float:
        """
        Calculate the demixed parameter of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The demixed parameter of the lattice.
        """
        params = []
        for frame in time_series:
            params.append(self.evaluate_frame(frame))
        self.quantity = np.array(params)

        return self.quantity

'''
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

    def evaluate_frame(self, frame: np.ndarray) -> float:
        """
        Calculate the magnetization of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The magnetization of the lattice.
        """
        # Magnetization is the mean value of the spins
        return abs(np.mean(frame))

    def evaluate(self, time_series: np.ndarray) -> float:
        """
        Calculate the magnetization of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The magnetization of the lattice.
        """
        magnetizations = []
        for frame in time_series:
            magnetizations.append(self.evaluate_frame(frame))
        self.quantity = np.array(magnetizations)

        return self.quantity
'''