# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


cov = np.array([[1, 0.5],
                [0.5, 1]])

def sample_p_x1_given_x2(x_2, mu, cov):
    mu1_given_2 = mu[0] + cov[0, 1] / cov[1, 1] * (x_2 - mu[1])
    cov1_given_2 = cov[1, 1] - cov[0, 1]  * cov[0, 1] / cov[0, 0]
    return np.random.normal(mu1_given_2, cov1_given_2)

def sample_p_x2_given_x1(x_1, mu, cov):
    mu2_given_1 = mu[1] + cov[1, 0] / cov[0, 0] * (x_1 - mu[0])
    cov2_given_1 = cov[0, 0] - cov[1, 0] * cov[1, 0] / cov[1, 1]
    return np.random.normal(mu2_given_1, cov2_given_1)


def gibbs_sampling(iteration=1000, cov=None):
    # randomly pick x_1 & x_2
    samples = np.zeros(shape=(iteration, 2))

    x_1 = 10
    x_2 = 4
    mu = np.array([x_1, x_2])

    samples[0, 0] = x_1
    samples[0, 1] = x_2
    for i in range(iteration-1):
        x_1 = sample_p_x1_given_x2(x_2, mu, cov)
        x_2 = sample_p_x2_given_x1(x_1, mu, cov)
        samples[i, 0] = x_1
        samples[i, 1] = x_2

    return samples

samples = gibbs_sampling(cov=cov)
# s = multivariate_normal.rvs(mu, cov, size=10000)
sns.set_context("paper")
sns.set_style("whitegrid")
sns.jointplot(samples[:,0], samples[:, 1], xlim=(0, 20), ylim=(0, 20), joint_kws={'s':1})


plt.show()
