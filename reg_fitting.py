import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling.fitting import LevMarLSQFitter

from dust_extinction.dust_extinction import GCC09_MWAvg, FM90

# get an observed extinction curve to fit
gcc09_model = GCC09_MWAvg()

x = gcc09_model.obsdata_x
# convert to E(x-V)/E(B0V)
y = (gcc09_model.obsdata_axav - 1.0)*gcc09_model.Rv
# only fit the UV portion (FM90 only valid in UV)
gindxs, = np.where(x > 3.125)

# initialize the model
fm90_init = FM90()

# pick the fitter
fit = LevMarLSQFitter()

# fit the data to the FM90 model using the fitter
#   use the initialized model as the starting point
gcc09_fit = fit(fm90_init, x[gindxs], y[gindxs])

print(gcc09_fit)

# plot the observed data, initial guess, and final fit
fig, ax = plt.subplots()

ax.plot(x, y, 'ko', label='Observed Curve')
ax.plot(x[gindxs], fm90_init(x[gindxs]), label='Initial guess')
ax.plot(x[gindxs], gcc09_fit(x[gindxs]), label='Fitted model')

ax.set_xlabel('$x$ [$\mu m^{-1}$]')
ax.set_ylabel('$E(x-V)/E(B-V)$')

ax.set_title('Example FM90 Fit to GCC09_MWAvg curve')

ax.legend(loc='best')
plt.tight_layout()
plt.show()
