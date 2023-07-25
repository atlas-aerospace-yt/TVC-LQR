"""
A python state space simulation.

Socials:
    https://github.com/atlas-aerospace-yt
    https://www.youtube.com/channel/UCWd6oqc8nbL-EX3Cxxk8wFA

23/07/2023
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Time constraints
DT = 0.001 # delta time s
RUN_TIME = 5 # simulation time s

# Hardware constants
F = 18.0 # average thrust N
I = 0.088 # mass moment of inertia kg.m^2
D = 0.45 # distance from TVC tocenter of mass

# Graphin place-holders
GraphX = []
GraphY = []

# State space begining values and setpoint
THETA_X = -4 * np.pi / 180
THETA_DOT_X = 0 * np.pi / 180
THETA_Y = 3 * np.pi / 180
THETA_DOT_Y = 0 * np.pi / 180

# State Space matrices
A = np.matrix([[0, 1, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 0]]) # constant state matrix

B = np.matrix([[0, 0],
               [F*D/I, 0],
               [0, 0],
               [0, F*D/I]]) # constant input matrix

Q = np.matrix([[10, 0, 0, 0],
               [0, 0.5, 0, 0],
               [0, 0, 10, 0],
               [0, 0, 0, 0.5]]) # "stabalise the system"

R = np.matrix([[0.1, 0],
               [0, 0.1]]) # "cost of energy to the system"

x = np.matrix([[THETA_X],
               [THETA_DOT_X],
               [THETA_Y],
               [THETA_DOT_Y]]) # state vector matrix

xf = np.matrix([[0],
                [0],
                [0],
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

    return k, y_prime, eig_vals

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

    GraphOutputX = []
    GraphOutputY = []

    # main loop
    for t in range(int(RUN_TIME / DT)):

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
        GraphOutputX.append(float(x[0]) * 180 / np.pi)
        GraphOutputY.append(float(x[2]) * 180 / np.pi)

    # appends plots to be plotted
    GraphY.append(GraphOutputX)
    GraphY.append(GraphOutputY)

    # outputs optimal gain
    print(K.round(3))

    # plots the matplotlib graph
    for item in GraphY:
        plt.plot(GraphX, item)

    plt.xlabel('time (s)')
    plt.ylabel('output (deg)')
    plt.title('State Space Output')
    plt.show()
