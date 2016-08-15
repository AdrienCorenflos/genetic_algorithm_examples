from genetic_algorithm import *


class Max2DFunction(GeneticAlgorithm):
    def __init__(self, constraints, majoring_square, fun, penalisation_factor=100., **kwords):
        super(Max2DFunction, self).__init__(**kwords)
        self.constraints = constraints
        self.fun = fun
        self.majoring_square = majoring_square
        self.penalisation_factor = penalisation_factor

    def individual(self):
        a = random.uniform()
        b = 1. - a
        return [a, b]

    def target(self):
        return None

    def fitness(self, individual, target):
        a = individual[0]
        b = individual[1]

        penalisation = self.penalisation_factor * abs(a + b - 1.)

        return self.fun(individual) - penalisation

    def mutate(self):
        for individual in self.fit:
            if self.mutation_prob > random.random():
                pos_to_mutate = random.randint(0, 2)
                individual[pos_to_mutate] = random.uniform(0, 1)
                _sum = np.sum(individual)
                for coordinate in range(len(individual)):
                    individual[coordinate] /= _sum

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
            a = male[0]
            b = female[1]

            child = [a / (a + b), b / (a + b)]
            children.append(child)
        self.fit.extend(children)
        self.population = self.fit[:]



def example():
    max_2d = Max2DFunction("constraints", ((-1, 1), (-1, 1)), lambda x: 1. - (x[0]) ** 2 - 0.75 * (x[1]) ** 2,
                           population_size=100, mutation_prob=0.05, pop_turn_over=0.2, diversification=0.05)

    print(max_2d.run_algorithm(until=10, verbose=False))

if __name__ == '__main__':
    example()