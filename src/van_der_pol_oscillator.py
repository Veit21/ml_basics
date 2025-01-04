import numpy as np
import matplotlib.pyplot as plt

# Define global variables
CARRYING_CAPACITY = 100
BIRTH_RATE = 1
N_0 = 1
DELTA_H = 0.05
T_0 = 0
T_END = 15
TIME_ARRAY = np.linspace(T_0, T_END, 100)
X_0 = 1
Y_0 = 1


def function_n(t):
    """Analytical solution to the Logistig Growth ODE.

    Args:
        t (numpy.ndarray): List of discrete time points at which the equation is to be evaluated.

    Returns:
        numpy.ndarray: Function values at time points t.
    """

    n_t = (N_0 * CARRYING_CAPACITY * np.exp(BIRTH_RATE * t)) / (CARRYING_CAPACITY + N_0 * (np.exp(BIRTH_RATE * t) - 1))
    return n_t

def n_prime(n_t):
    """Differential equation for logistic growth

    Args:
        n_t (float): Number of individuals at time t.

    Returns:
        float: dn/dt
    """

    return BIRTH_RATE * n_t * (1 - n_t/CARRYING_CAPACITY)

def runge_kutta(function, t_0, x_0, y_0, d_h, t_end=T_END):
    """Runge-Kutta 4 method to integrate an ODE.

    Args:
        function (_type_): _description_
        t_0 (float): Starting time. Lower bound of integration.
        n_0 (float): Starting number of N. 
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

def runge_kutta_alt(function, t_0, n_0, d_h, t_end=T_END):
    """Runge-Kutta 4 method to integrate an ODE.

    Args:
        function (_type_): _description_
        t_0 (float): Starting time. Lower bound of integration.
        n_0 (float): Starting number of N. 
        d_h (float): Integration interval. Step width.
        t_end (float, optional): End time. Upper bound of integration. Defaults to T_END.

    Returns:
        tuple: Time points and approximated function value at time points t.
    """
    n = n_0
    steps = np.arange(t_0, t_end + d_h, d_h)
    n_approx = [n]
    for t in steps:
        k1 = function(n)
        k2 = function(n + k1 * d_h/2)
        k3 = function(n + k2 * d_h/2)
        k4 = function(n + k3 * d_h)
        k_array = np.array([k1, k2, k3, k4])
        weights = np.array([1/6, 1/3, 1/3, 1/6])
        n += d_h * np.dot(weights, k_array)
        n_approx.append(n)
    return steps, n_approx[:-1]

# TODO: Somehow make parameter mu variable.
def vdp_prime(x, y, mu=2.0):
    """Van der Pol oscillator. System of two interacting equations.

    Args:
        x (float): Current x position.
        y (float): Current y position.
        mu (int, optional): Damping parameter. Defaults to 2.

    Returns:
        tuple: Rates of change (dx/dt, dy/dt).
    """

    F_x = 1/3 * x**3 - x
    x_prime = mu * (y - F_x)
    y_prime = -(1/mu)*x
    return x_prime, y_prime

if __name__ == '__main__':

    # Analytical solution to the ODE
    N_T = function_n(TIME_ARRAY)

    # Runge-Kutta 4 solution to the ODE
    rk4_steps, N_t_rk4 = runge_kutta_alt(lambda n: n_prime(n), T_0, N_0, DELTA_H)

    # Plot
    """
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1, 1, 1)
    axs.plot(TIME_ARRAY, N_T, c='k', linewidth=1.0, label='Analytic')
    axs.scatter(rk4_steps, N_t_rk4, c='m', s=4, label='Runge-Kutta')
    axs.set_xlabel('t')
    axs.set_ylabel('N(t)')
    axs.set_title('Logistic Growth')
    axs.legend()
    axs.grid()
    plt.show()
    """

    # Initial conditions
    X_0 = 0.0
    Y_0 = 0.0

    # Now try that with the Van der Pol oscillator
    rk4_vdp_steps, x_t_vdp, y_t_vdp = runge_kutta(lambda x, y: vdp_prime(x, y, mu=2.0), T_0, X_0, Y_0, DELTA_H, t_end=100)

    # Plot
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1, 1, 1)
    axs.plot(x_t_vdp, y_t_vdp, c='m')
    axs.set_xlabel('x(t)')
    axs.set_ylabel('y(t)')
    axs.set_title('Van der Pol oscillator')
    axs.grid()
    plt.show()
