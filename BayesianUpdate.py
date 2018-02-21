# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm

fig, ax = plt.subplots(1, 1)

mean = 3.5
variance = 0.1

x = np.linspace(0.0, 10.0, 1000)

ground_truth = norm(loc=5, scale=math.sqrt(1.0))
prior = norm(loc=mean, scale=math.sqrt(variance))


def calculate_posterior(prior_mean, prior_variance, observations, known_variance):
    posterior_mean = math.pow(
        (1.0 / prior_variance) + (observations.size / known_variance), -1)
    posterior_mean = posterior_mean * \
        ((prior_mean / prior_variance) + (np.sum(observations) / known_variance))

    posterior_variance = (known_variance * prior_variance) / \
        (known_variance + observations.size * prior_variance)

    return posterior_mean, posterior_variance


sample_size = 100
sample_list = []
for n in range(sample_size):
    sample = ground_truth.rvs(size=1)
    sample_list.append(sample)
    samples = np.array(sample_list)
    p_mean, p_variance = calculate_posterior(
        prior.mean(), prior.var(), samples, ground_truth.var())
    posterior_est = norm(loc=p_mean, scale=math.sqrt(p_variance))
    if n == sample_size-1:
        ax.hist(samples, normed=True, histtype='stepfilled',
                alpha=0.2, label="random draws")
        ax.plot(x, posterior_est.pdf(x), 'y', lw=2, alpha=n / sample_size,
                label="posterior N(%.2f, %.2f)" % (posterior_est.mean(),
                                                   posterior_est.var()))
    else:
        ax.plot(x, posterior_est.pdf(x), 'y', lw=2, alpha=n / sample_size)

print("Stats:")
print(prior.stats(moments="mv"))

ax.plot(x, prior.pdf(x), 'r-', lw=2, label="prior N(%.1f, %.1f)" %
        (prior.mean(), prior.var()))
ax.plot(x, ground_truth.pdf(x), 'g-', lw=2,
        label="ground truth N(%.1f, %.1f)" % (ground_truth.mean(), ground_truth.var()))

ax.axvline(ground_truth.mean(), color='b', label="actual mean %.2f" % ground_truth.mean(), linestyle='dashed',
           linewidth=2)
ax.legend(loc='best', frameon=False)
plt.show()
