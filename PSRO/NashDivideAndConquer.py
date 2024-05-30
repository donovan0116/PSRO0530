import numpy as np


class NashDivideAndConquer:
    def __init__(self, population, payoffs):
        self.population = population
        self.payoffs = payoffs
        self.num_actors = len(population)

    def solve(self, actors_indices):
        if len(actors_indices) == 1:
            return np.array([1.0])
        mid = len(actors_indices) // 2
        left_prob = self.solve(actors_indices[:mid])
        right_prob = self.solve(actors_indices[mid:])
        left_payoff = sum(self.payoffs[actors_indices[i]][actors_indices[j]] *
                          left_prob[i]
                          for i in range(len(left_prob))
                          for j in range(len(right_prob)))
        right_payoff = sum(self.payoffs[actors_indices[i]][actors_indices[j]] *
                           right_prob[i]
                           for i in range(len(right_prob))
                           for j in range(len(left_prob)))
        total_payoff = left_payoff + right_payoff
        probabilities = np.concatenate([left_prob * (left_payoff / total_payoff),
                                        right_prob * (right_payoff / total_payoff)])
        return probabilities

    def get_nash_probabilities(self):
        actors_indices = list(range(self.num_actors))
        return self.solve(actors_indices)
# 示例用法
# nash_dc = NashDivideAndConquer(actor_pop, payoff)
# sample_proportion = nash_dc.get_nash_probabilities()
