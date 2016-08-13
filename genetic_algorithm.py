import numpy as np
from numpy import random


class GeneticAlgorithm(object):
    def __init__(self, population_size, pop_turn_over=0.2,
                 diversification=0.05, mutation_prob=0.01,
                 seed=1):

        """
        This is how we define a genetic algorithm, independently of its methods

        :param population_size: defines the number of individuals available for test at
            each generation
        :param pop_turn_over: the turnover in the population at each generation (death rate)
        :param diversification: the number of non fit individuals that will be saved by chance
        :mutation_prob: the probability for one individual to mutate

        """
        self.population_size = population_size
        self.pop_turn_over = pop_turn_over
        self.mutation_prob = mutation_prob
        self.diversification = diversification

        self.population = None
        """This is to distinguish from the fit individuals and those who aren't"""
        self.fit = None
        self.non_fit = None

        random.seed(seed)

    def individual(self):
        """
        Depending on your implementation:
        How do you create a random initial individual?
        """
        raise NotImplementedError

    def fitness(self, individual, target):
        """
        How do you measure the fitness of your individual given your target
        """
        raise NotImplementedError

    def target(self):
        """
        Create a target
        """
        raise NotImplementedError

    def select(self):
        """
        This is how you define the fittest selection
        """
        evaluated = [(self.fitness(x, self.target), x) for x in self.population]
        evaluated = [individual[1] for individual in sorted(evaluated)]

        index = int(self.pop_turn_over * len(evaluated))
        selected, non_selected = evaluated[:index], evaluated[index:]

        self.fit = selected
        self.non_fit = non_selected

    def diversify(self):
        """
        Add some lucky unfit individuals
        """
        lucky_ones = [individual for individual in self.non_fit if self.diversification > random.random()]
        self.fit.extend(lucky_ones)

    def mutate(self):
        """
        How do you mutate an individual?
        """
        raise NotImplementedError

    def mate(self):
        """
        In this version parents are kept alive by the end of the process.
        """
        parents_length = len(self.fit)
        full_replacement = self.population_size - parents_length
        children = []
        for child in range(full_replacement):
            male, female = random.choice(len(self.fit), 2, replace=False)
            male = self.fit[male]
            female = self.fit[female]
            half = len(male) // 2
            child = male[:half] + female[half:]
            children.append(child)
        self.fit.extend(children)
        self.population = self.fit[:]

    def evolve(self):
        """
        One evolution round
        """
        self.select()
        self.diversify()
        self.mutate()
        self.mate()

    def first_breed(self):
        self.population = [self.individual() for _ in range(self.population_size)]

    def print_convergence(self):
        """
        Method for verbose
        """
        intermediate = [(self.fitness(x, self.target), x) for x in self.population]
        print(np.mean([x[0] for x in intermediate]))

    def run_algorithm(self, until=30, verbose=True):
        self.first_breed()

        for _ in range(until):
            self.evolve()
            if verbose:
                self.print_convergence()

        result = [(self.fitness(x, self.target), x) for x in self.population]
        return sorted(result)
