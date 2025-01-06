import numpy as np
import matplotlib.pyplot as plt

def runge_kutta(function, t_0, x_0, y_0, d_h, t_end=25.0):
    """Runge-Kutta 4 method to integrate an ODE.

    Args:
        function (_type_): _description_
        t_0 (float): Starting time. Lower bound of integration.
        d_h (float): Integration interval. Step width.
        t_end (float, optional): End time. Upper bound of integration. Defaults to T_END.

    Returns:
        list: Time points and approximated function values (x, y) at time points t.
    """
    
    # Initial condition
    x = x_0
    y = y_0
    
    # Simulate over discrete time steps
    steps = np.arange(t_0, t_end + d_h, d_h)
    
    # Keep track of the (x,y) positions
    x_approx = [x]
    y_approx = [y]

    # Iterate over time steps. Simulate both for x and y!
    for t in steps:
        k1_x, k1_y = function(x, y)
        k2_x, k2_y = function(x + k1_x * d_h/2, y + k1_y * d_h/2)
        k3_x, k3_y = function(x + k2_x * d_h/2, y + k2_y * d_h/2)
        k4_x, k4_y = function(x + k3_x * d_h, y + k3_y * d_h)
        k_x_array = np.array([k1_x, k2_x, k3_x, k4_x])
        k_y_array = np.array([k1_y, k2_y, k3_y, k4_y])
        weights = np.array([1/6, 1/3, 1/3, 1/6])
        x += d_h * np.dot(weights, k_x_array)
        y += d_h + np.dot(weights, k_y_array)
        x_approx.append(x)
        y_approx.append(y)

    return steps, x_approx[:-1], y_approx[:-1]

def predator_prey_prime(x, y):
    """Simple predator prey system. System of two interacting equations.

    Args:
        x (float): Current x position.
        y (float): Current y position.
        mu (int, optional): Damping parameter. Defaults to 2.

    Returns:
        tuple: Rates of change (dx/dt, dy/dt).
    """

    x_dot = (3000 - x - 4 * y) * x
    y_dot = (1000 - (x/2) - y) * y
    return x_dot, y_dot


if __name__ == '__main__':

    # Initial conditions
    X_0 = 3000.0
    Y_0 = 1000.0
    T_0 = 0.0
    DELTA_H = 0.01

    # Get the gradient at given (X, Y) positions
    NUM_STEPS = 30
    x_coord = np.linspace(0.0, 3100.0, NUM_STEPS).astype(np.float128)
    y_coord = np.linspace(0.0, 1100.0, NUM_STEPS).astype(np.float128)

    X, Y = np.meshgrid(x_coord, y_coord)
    grad_x, grad_y = predator_prey_prime(X, Y)
    vec_color = np.sqrt(np.square(grad_y) + np.square(grad_y))  # Color the arrows according to their magnitude

    # Plot vector field
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1, 1, 1)
    axs.quiver(X, Y, grad_x, grad_y, vec_color)    # 'vectors' variable can be accessed and modified later
    axs.set_xlabel('x(t)')
    axs.set_ylabel('y(t)')
    axs.set_title('Predator-Prey System - Interactive')
    axs.set_xlim([0.0, 3100.0])
    axs.set_ylim([0.0, 1100.0])
    axs.grid()

    ## In a second plot, show superimposed dynamics with altering initial conditions
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1,1,1)
    axs.set_title("Predator Prey model - Phase plot")
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.set_xlim([0.0, 3100.0])
    axs.set_ylim([0.0, 1100.0])
    axs.grid()

    NUM_STEPS = 10
    x_coord = np.linspace(0.0, 3100.0, NUM_STEPS).astype(np.float128)
    y_coord = np.linspace(0.0, 1100.0, NUM_STEPS).astype(np.float128)

    # Iterate over all x coordinates
    for i in x_coord:

        # Iterate over all y coordinates
        for j in y_coord:

            # Evaluate ODEs at every possible (x_0, y_0)
            rk4_vdp_steps, x_t, y_t = runge_kutta(lambda x, y: predator_prey_prime(x, y), T_0, i, j, DELTA_H, t_end=250.0)

            # Plot the dynamics
            axs.plot(x_t, y_t, c='k', linewidth=0.2, alpha=0.65)

    # Show plot
    plt.show()

    # NOTE: Simulation does not quite work yet. Probably no good parametercombinations.