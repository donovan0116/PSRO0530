import numpy as np


class NashDP:
    def __init__(self, population, payoffs):
        self.population = population
        self.payoffs = payoffs  # payoff矩阵
        self.num_actors = len(population)
        self.dp_table = {}

    def value_iteration(self, gamma=0.99, epsilon=1e-6):
        for i in range(self.num_actors):
            self.dp_table[i] = np.zeros(self.num_actors)

        while True:
            delta = 0
            for i in range(self.num_actors):
                v = self.dp_table[i].copy()
                for j in range(self.num_actors):
                    max_value = float('-inf')
                    for k in range(self.num_actors):
                        value = self.payoffs[i][k] + gamma * self.dp_table[k].sum()
                        if value > max_value:
                            max_value = value
                    self.dp_table[i][j] = max_value
                delta = max(delta, np.abs(v - self.dp_table[i]).sum())
            if delta < epsilon:
                break

    def get_nash_probabilities(self):
        probabilities = np.zeros(self.num_actors)
        for i in range(self.num_actors):
            probabilities[i] = self.dp_table[i].sum()
        return probabilities / probabilities.sum()

# 示例用法
# nash_dp = NashDP(actor_pop, payoff)
# nash_dp.value_iteration()
# sample_proportion = nash_dp.get_nash_probabilities()
