# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm

fig, ax = plt.subplots(1, 1)

mean = -1.0
variance = 0.2

x = np.linspace(-2.5, 10.0, 1000)

ground_truth = norm(loc=5, scale=math.sqrt(1.0))
prior = norm(loc=mean, scale=math.sqrt(variance))


def calculate_posterior(prior_mean, prior_variance, observations, known_variance):
    posterior_mean = math.pow((1.0 / prior_variance) + (observations.size / known_variance), -1)
    posterior_mean = posterior_mean * ((prior_mean / prior_variance) + (np.sum(observations) / known_variance))

    posterior_variance = (known_variance * prior_variance) / (known_variance + observations.size * prior_variance)

    return posterior_mean, posterior_variance


for n in range(30):

    sample_size = n
    samples = ground_truth.rvs(size=sample_size)

    p_mean, p_variance = calculate_posterior(prior.mean(), prior.var(), samples, ground_truth.var())
    posterior_est = norm(loc=p_mean, scale=math.sqrt(p_variance))
    if n == 29:
        ax.hist(samples, normed=True, histtype='stepfilled', alpha=0.2, label="random draws")
        ax.plot(x, posterior_est.pdf(x), 'y', lw=2, alpha=n / 45.0,
                label="posterior N(%.2f, %.2f)" % (posterior_est.mean(),
                                                   posterior_est.var()))
    else:
        ax.plot(x, posterior_est.pdf(x), 'y', lw=2, alpha=n / 45.0)

print("Stats:")
print(prior.stats(moments="mv"))

ax.plot(x, prior.pdf(x), 'r-', lw=2, label="prior N(%.1f, %.1f)" % (prior.mean(), prior.var()))
ax.plot(x, ground_truth.pdf(x), 'g-', lw=2,
        label="ground truth N(%.1f, %.1f)" % (ground_truth.mean(), ground_truth.var()))

ax.axvline(ground_truth.mean(), color='b', label="actual mean %.2f" % ground_truth.mean(), linestyle='dashed',
           linewidth=2)
ax.legend(loc='best', frameon=False)
plt.show()
