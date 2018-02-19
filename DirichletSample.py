# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import dirichlet, multinomial


# dirichlet - HABIT's evaluation
alpha = np.ones(5)
alpha = dirichlet(alpha).rvs(size=1)[0] * 7



samples = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 66, 80, 100]
x = np.array(samples)
y = np.zeros(len(samples))
err = np.zeros(len(samples))

runs = 1000

for n in range(len(samples)):

    y_s = np.zeros(runs)

    for r in range(runs):
        print("Run #",  r + (n * runs))
        truster = dirichlet(alpha).rvs(size=1)[0]

        alpha_p = np.ones(5)

        for s in range(samples[n]):
            alpha_p += multinomial(1, truster).rvs(size=1)[0]

        y_s[r] = np.sum(np.abs(truster - alpha_p / np.sum(alpha_p)))

    y[n] = np.mean(y_s)
    err[n] = np.std(y_s)


plt.plot(x, y, 'r-', lw=2, label="DRSTruster")
plt.errorbar(x, y, yerr=2 * err)

plt.legend(loc='best', frameon=False)
plt.show()
