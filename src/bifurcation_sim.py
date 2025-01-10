import numpy as np
import matplotlib.pyplot as plt

def fn(z_n, theta):
    """Simple nonlinear iteration. 'Difference equation'.

    Args:
        z_n (float): Function value z at iteration n.
        theta (float): Bifurcation parameter.

    Returns:
        float: Function value at iteration n+1.
    """
    return z_n**2 + theta

def logistic_map(x_n, r):
    """Logistic map

    Args:
        x_n (float): Current function value at iteration n.
        r (float): Bifurcation parameter.

    Returns:
        float: Function value at iteration n+1.
    """
    return r*x_n*(1-x_n)

if __name__ == '__main__':

    # Number of simulations
    N = 10000

    # Number of iterations per simulation
    num_iterations = 200

    # Initial conditions
    z = np.ones(N)

    # Parameter space
    theta = np.linspace(-2.0, 0.6, N)

    # Prepare a diagram
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1, 1, 1)
    axs.set_xlabel('r')
    axs.set_ylabel('x')
    axs.set_title('Nonlinear iteration - Bifurcation diagram')
    axs.set_xlim(-2., 0.5)
    axs.grid()

    # Simulate for each r
    for i in range(num_iterations):
        z = fn(z, theta)

        if i >= (num_iterations - 50):
            axs.plot(theta, z, ',k', alpha=.15)
    
    # Show plot
    plt.show()