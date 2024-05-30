import numpy as np


class NashGreedy:
    def __init__(self, population, payoffs):
        self.population = population
        self.payoffs = payoffs
        self.num_actors = len(population)

    def get_nash_probabilities(self):
        probabilities = np.zeros(self.num_actors)
        for i in range(self.num_actors):
            best_response = np.argmax(self.payoffs[i])
            probabilities[best_response] += 1
        return probabilities / probabilities.sum()
# 示例用法
# nash_greedy = NashGreedy(actor_pop, payoff)
# sample_proportion = nash_greedy.get_nash_probabilities()
