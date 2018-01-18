# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

# prior
a, b = 10, 2

fig, ax = plt.subplots(1, 1)

x = np.linspace(0, 1.0, 1000)
beta_distribution = beta(a, b)
ax.plot(x, beta_distribution.pdf(x), 'k-', lw=2, label="prior beta dist Beta(%s, %s)" % (a, b))


class DirichletProcess:
    def __init__(self, base_distribution, concentration):
        self.concentration = concentration
        self.base_distribution = base_distribution
        self.n = 0  # number of draws
        self.x_s = {}

    def draw(self):
        x_n = None
        if self.n == 0:
            # make a draw from base distribution
            x_n = self.base_distribution.rvs()
        else:
            choices = [i for i in self.x_s.keys()]
            probabilities = self.calculate_probabilities(choices)
            x_n = np.random.choice(choices, replace=True, p=probabilities)
            if x_n == "base":
                x_n = self.base_distribution.rvs()
            else:
                x_n = float(x_n)

        x_n = np.around(x_n, decimals=2)
        self.record(x_n)
        self.n += 1
        return x_n

    def record(self, draw):
        print(draw)
        if draw not in self.x_s:
            self.x_s[draw] = 0
        self.x_s[draw] += 1

    def calculate_probabilities(self, choices):
        probabilities = np.zeros(len(choices) + 1)
        denominator = (self.concentration + self.n)
        probabilities.itemset(0, self.concentration / denominator)
        index = 1
        for choice in choices:
            probabilities.itemset(index, self.x_s[choice] / denominator)
            index += 1

        choices.insert(0, "base")
        print(choices)
        print(probabilities)
        print("probabilities", np.sum(probabilities))

        return probabilities

    def rvs(self, size=1):
        if size == 1:
            return self.draw()
        else:
            return np.array([self.draw() for _ in range(size)])


concentration = 100.0
dirichletProcess = DirichletProcess(beta_distribution, concentration)

dp_samples = dirichletProcess.rvs(100)
samples = beta_distribution.rvs(size=100)

ax.hist(samples, normed=True, histtype='stepfilled', alpha=0.2, label="sample")
ax.hist(dp_samples, normed=True, histtype='stepfilled', alpha=0.2, label="DP(a=%s, H)" % concentration)
ax.legend(loc='best', frameon=False)
plt.show()
