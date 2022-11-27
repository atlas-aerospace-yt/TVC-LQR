"""
A python state space simulation.

Socials:
    https://github.com/atlas-aerospace-yt
    https://www.youtube.com/channel/UCWd6oqc8nbL-EX3Cxxk8wFA

27/11/2022
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


# Hardware constants
F = 18.0 # average thrust N
I = 0.088 # mass moment of inertia kg.m^2
D = 0.45 # distance from TVC tocenter of mass
DT = 0.001 # delta time s

# Graphin place-holders
GraphX = []
GraphY = []

# State space begining values and setpoint
THETA = 0.0 # angle of the rocket rad
THETA_DOT = 0.07 # velocity of the rocket rad

# State Space matrices
A = np.matrix([[0, 1],
               [0, 0]]) # constant state matrix

B = np.matrix([[0],
               [F*D/I]]) # constant input matrix

Q = np.matrix([[0.5, 0],
               [0, 0.005]]) # "stabalise the system"

R = np.matrix([[1]]) # "cost of energy to the system"

x = np.matrix([[THETA],
               [THETA_DOT]]) # state vector matrix

xf = np.matrix([[0],
                [0]]) # setpoint matrix

def lqr():
    """
    Returns the optimal values and eigen values for the simulation

    Returns:
        np.array: the optimal results
        np.array: the riccati equation
        np.array: the eigen values
    """

    # solves algebraic riccati equation
    y_prime = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # computes the optimal K value
    k = np.matrix(scipy.linalg.inv(R)*(B.T*y_prime))

    # compute the eigenvalues
    eig_vals = np.linalg.eigvals(A-np.dot(B,k))

    return k, x, y_prime

# State function
def update_state():
    """
    Updates the state space model - discrete
    This output must then be used in the equation x = x + dt * updated state

    Returns:
        np.array: the change in state
    """

    # updates the state
    x_dot = A * x + B * u

    return x_dot

if __name__ == "__main__":

    # main loop
    for t in range(2000):

        # gets optimal gain (K)
        K, S, E = lqr()

        # calculates the error from setpoint
        e = x - xf

        # calculates optimal output (u)
        u = -K * e

        # updates the state (x)
        x = x + DT * update_state()

        # appending the graph
        GraphX.append(t * DT)
        GraphY.append(float(x[1]) * 180 / np.pi)

    # outputs optimal gain
    print(K)

    # plots the matplotlib graph
    plt.plot(GraphX, GraphY)
    plt.xlabel('time (s)')
    plt.ylabel('output (deg)')
    plt.title('State Space Output')
    plt.show()
