import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Generate the time series data from the van der Pol oscillator
mu = 1.0
t_span = [0, 100]
y0 = [0.0, 1.0]

def vanderpol(t, y, mu):
    dydt = [y[1], mu*(1 - y[0]**2)*y[1] - y[0]]
    return dydt

sol = solve_ivp(lambda t, y: vanderpol(t, y, mu), t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], num=5001))
Y = sol.y.T

# Embed the time series data in a higher dimensional space using time delay embedding
m = 2  # number of components of the state vector
tau = 1.0  # time delay
n = 3  # embedding dimension
N = len(Y) - (n - 1) * tau  # number of time delay vectors

Y_embed = np.zeros((N, n * m))

for i in range(N):
    Y_embed[i] = np.concatenate([Y[i + j * tau] for j in range(n)])

# Estimate the derivatives of the time delay vectors using finite differences
h = 1  # finite difference step size

Y_dot = np.zeros((N, n * m))

for i in range(N):
    y = Y_embed[i]
    yh = Y_embed[i + h] if i + h < N else Y_embed[i]
    Y_dot[i] = (yh - y) / h

# Solve for the coefficients of the dynamical system using linear regression
X = Y_embed[:, :n*m]
Y = Y_dot[:, 2*m:]

A = np.linalg.lstsq(X, Y, rcond=None)[0]

# Extract the coefficients of the van der Pol oscillator
C = A[:m*n].reshape((m, n))
f = lambda y: np.concatenate([y[m:], mu * (1 - y[:m]**2) * y[m:] - y[:m]])
J = lambda y: np.array([[0, 1], [-2 * mu * y[0] * y[1] - 1, mu * (1 - y[0]**2)]])

# Evaluate the reconstructed system and compare to the original
Y_recon = np.zeros(Y.shape)
Y_recon[0] = Y[0]

for i in range(1, N):
    y = Y_recon[i-1]
    y_next = y + f(y) * tau
    Y_recon[i] = y_next

plt.plot(sol.t, sol.y[0], label='Original')
plt.plot(sol.t[:N], Y_recon[:, 0], label='Reconstructed')
plt.legend()
plt.show()

plt.plot(sol.y[0], sol.y[1], label='Original')
plt.plot(Y_recon[:, 0], Y_recon[:, 1], label='Reconstructed')
plt.legend()
plt.show()
