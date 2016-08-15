import numpy as np
from numpy import random
from genetic_algorithm import *


class TargetSumGenetic(GeneticAlgorithm):
    """
    Class to target the sum of components over a vector

    """

    def __init__(self, min_value, max_value, nb_dimensions, target_value=300,
                 **kwargs):
        super(TargetSumGenetic, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.nb_dimensions = nb_dimensions
        self.target_value = target_value

    def individual(self):
        return [random.randint(self.min_value, self.max_value)
                for _ in range(self.nb_dimensions)]

    def fitness(self, individual, target):
        return np.abs(np.sum(individual) - target())

    def target(self):
        return self.target_value

    def mutate(self):
        for individual in self.fit:
            if self.mutation_prob > random.random():
                pos_to_mutate = random.randint(0, self.nb_dimensions)
                individual[pos_to_mutate] = random.randint(self.min_value, self.max_value)

