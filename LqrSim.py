import random
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Hardware constants
F = 18.0 # average thrust N
I = 0.1 # mass moment of inertia kg.m^2
D = 0.45 # distance from TVC to flight computer m
DT = 0.001 # delta time s

# Graphin place-holders
GraphX = []
GraphY = []

# State space begining values and setpoint
theta = -0.15 # angle of the rocket rad
theta_dot = 0.15 # velocity of the rocket rad

# State Space matrices
A = np.matrix([[0, 1],
               [0, 0]]) # constant state matrix

B = np.matrix([[0],
               [F*D/I]]) # constant input matrix

Q = np.matrix([[10, 0],
               [0, 10]]) # "stabalise the system"

R = np.matrix([[0.1]]) # "cost of energy to the system"

x = np.matrix([[theta],
               [theta_dot]]) # state vector matrix

xf = np.matrix([[0],
                [0]]) # setpoint matrix

# LQR function
def Lqr(A,B,Q,R):

    # solves algebraic riccati equation
    P = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # computes the optimal K value
    K = np.matrix(scipy.linalg.inv(R)*(B.T*P))

    return -K

# State function
def UpdateState(A, x, B, u):

    # updates the state
    x_dot = A * x + B * u

    return x_dot

if __name__ == "__main__":

    # main loop
    for t in range(2000):

        # gets optimal gain (K)
        K = Lqr(A, B, Q, R)

        # calculates the error from setpoint
        e = x - xf

        # calculates optimal output (u)
        u = K * e

        # updates the state (x)
        x = x + DT * UpdateState(A,x,B,u)

        # appending the graph
        GraphX.append(t * DT)
        GraphY.append(float(x[0]) * 180 / np.pi)

    # outputs optimal gain
    print(K)

    # plots the matplotlib graph
    plt.plot(GraphX, GraphY)
    plt.xlabel('time (s)')
    plt.ylabel('output (deg)')
    plt.title('State Space Output')
    plt.show()
