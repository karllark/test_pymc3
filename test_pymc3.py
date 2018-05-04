import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

# following command needed for linux
# $ export MKL_THREADING_LAYER=GNU
import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

# get maximum likelihood result

map_estimate = pm.find_MAP(model=basic_model)

print(map_estimate)

# now get the NUTS sampler

from scipy import optimize

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500)

print(trace['alpha'][-5:])
print(trace['beta'][-5:])

print(pm.summary(trace))

pm.traceplot(trace)
