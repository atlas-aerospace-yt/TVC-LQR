import os
try:
    import random
    import numpy as np
    import scipy.linalg
    import matplotlib.pyplot as plt
except:
    os.system("pip install matplotlib")
    os.system("pip install numpy")
    os.system("pip install scipy")

# Hardware constants
F = 18.0 # average thrust N
I = 0.088 # mass moment of inertia kg.m^2
D = 0.45 # distance from TVC to flight computer m
DT = 0.001 # delta time s

# Graphin place-holders
GraphX = []
GraphY = []

# State space begining values and setpoint
theta = 0.0 # angle of the rocket rad
theta_dot = 0.07 # velocity of the rocket rad

# State Space matrices
A = np.matrix([[0, 1],
               [0, 0]]) # constant state matrix

B = np.matrix([[0],
               [F*D/I]]) # constant input matrix

Q = np.matrix([[0.5, 0],
               [0, 0.005]]) # "stabalise the system"

R = np.matrix([[1]]) # "cost of energy to the system"

x = np.matrix([[theta],
               [theta_dot]]) # state vector matrix

xf = np.matrix([[0],
                [0]]) # setpoint matrix

# LQR function
def Lqr(A,B,Q,R):

    # solves algebraic riccati equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # computes the optimal K value
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))

    # compute the eigenvalues
    S = np.linalg.eigvals(A-np.dot(B,K))

    return K, X, S

# State function
def UpdateState(A, x, B, u):

    # updates the state
    x_dot = A * x + B * u

    return x_dot

if __name__ == "__main__":

    # main loop
    for t in range(2000):

        # gets optimal gain (K)
        K, S, E = Lqr(A, B, Q, R)

        # calculates the error from setpoint
        e = x - xf

        # calculates optimal output (u)
        u = -K * e

        # updates the state (x)
        x = x + DT * UpdateState(A,x,B,u)

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

