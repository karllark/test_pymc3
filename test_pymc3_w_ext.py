import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

from dust_extinction.dust_extinction import GCC09_MWAvg, FM90

import theano
import theano.tensor as tt

def fm90func(c1, c2, c3, c4, xo, gamma, x, hs):
    return c1 + c2*x + c3*(x**2/((x**2 - xo**2)**2 + (x**2)*(gamma**2))) + hs*c4*(0.5392*((x - 5.9)**2) + 0.05644*((x - 5.9)**3))

class FM90theano(theano.Op):
    def make_node(self, M, e):
        c1 = tt.as_tensor_varible(c1)
        c2 = tt.as_tensor_varible(c2)
        c3 = tt.as_tensor_varible(c3)
        c4 = tt.as_tensor_varible(c4)
        xo = tt.as_tensor_varible(xo)
        gamma = tt.as_tensor_varible(gamma)
        return theano.Apply(self, [c1, c2, c3, c4, xo, gamma], [c1.type()])

    def perform(self, node, inputs, outputs):
        c1, c2, c3, c4, xo, gamma = inputs
        outputs[0][0] = fm90func(c1, c2, c3, c4, xo, gamma)

c1 + c2*x + c3*(x**2/((x**2 - xo**2)**2 + (x**2)*(gamma**2))) + hs*c4*(0.5392*((x - 5.9)**2) + 0.05644*((x - 5.9)**3))

    def grad(self, inputs, g):
        M, e = inputs
        E = self(M, e)
        dE_dM = 1. / (1.0 - e * tt.cos(E))
        dE_de = tt.sin(E) * dE_dM
        return dE_dM * g[0], (dE_de * g).sum()

    def infer_shape(self, node, i0_shapes):
        return [i0_shapes[0]]


# get an observed extinction curve to fit
gcc09_model = GCC09_MWAvg()

x_full = gcc09_model.obsdata_x
# convert to E(x-V)/E(B0V)
y_full = (gcc09_model.obsdata_axav - 1.0)*gcc09_model.Rv
y_unc_full = gcc09_model.obsdata_axav_unc*gcc09_model.Rv
# only fit the UV portion (FM90 only valid in UV)
gindxs, = np.where(x_full > 3.125)

# get the data to fit
x = x_full[gindxs]
y = y_full[gindxs]
y_unc = y_unc_full[gindxs]

# heavyside Function
hs = np.zeros((len(gindxs)))
gindxs, = np.where(x >= 5.9)
hs[gindxs] = 1.0

# following command needed for linux
# $ export MKL_THREADING_LAYER=GNU
import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))

basic_model = pm.Model()

with basic_model:

    # Priors for model parameters
    c1 = pm.Normal('C1', mu=1.0, sd=10.)
    c2 = pm.Normal('C2', mu=2.0, sd=10.)
    c3 = pm.Normal('C3', mu=1.23, sd=10.)
    c4 = pm.Normal('C4', mu=0.41, sd=10.)
    xo = pm.Normal('xo', mu=4.2, sd=1.)
    gamma = pm.Normal('gamma', mu=0.99, sd=1.)

    # model
    mu = c1 + c2*x + c3*(x**2/((x**2 - xo**2)**2 + (x**2)*(gamma**2))) + hs*c4*(0.5392*((x - 5.9)**2) + 0.05644*((x - 5.9)**3))

    # does not work
    #mu = fm90func(c1, c2, c3, c4, xo, gamma, x, hs)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=y_unc, observed=y)

# get maximum likelihood result

map_estimate = pm.find_MAP(model=basic_model)

print(map_estimate)

# now get the NUTS sampler

from scipy import optimize

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(1000)

print(pm.summary(trace))

pm.traceplot(trace)
