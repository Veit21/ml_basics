import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

def lorenz_equations(x, y, z, sigma, rho, beta):
    """Lorenz equations. A system of three interacting ODEs.

    Args:
        x (float): x coordinate.
        y (float): y coordinate.
        z (float): z coordinate.
        sigma (float): Parameter.
        rho (float): Parameter.
        beta (float): Parameter.

    Returns:
        tuple: Rates of change in x, z and y direction (dx, dy, dz).
    """

    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * y

    return x_dot, y_dot, z_dot

def runge_kutta(function, t_0, x_0, y_0, z_0, d_h, t_end):
    """Runge-Kutta 4 method to integrate an ODE.

    Args:
        function (function): Lambda function to be evaluated each step dt. The lambda function represents the ODE to be integrated.
        t_0 (float): Starting time. Lower bound of integration.
        d_h (float): Integration interval. Step width.
        t_end (float, optional): End time. Upper bound of integration. Defaults to T_END.

    Returns:
        list: Time points and approximated function values (x, y) at time points t.
    """
    
    # Initial condition
    x = x_0
    y = y_0
    z = z_0
    
    # Simulate over discrete time steps
    steps = np.arange(t_0, t_end + d_h, d_h)
    
    # Keep track of the (x,y) positions
    x_approx = [x]
    y_approx = [y]
    z_approx = [z]

    # Iterate over time steps. Simulate both for x and y!
    for t in steps:
        k1_x, k1_y, k1_z = function(x, y, z)
        k2_x, k2_y, k2_z = function(x + k1_x * d_h/2, y + k1_y * d_h/2, z + k1_z * d_h/2)
        k3_x, k3_y, k3_z = function(x + k2_x * d_h/2, y + k2_y * d_h/2, z + k2_z * d_h/2)
        k4_x, k4_y, k4_z = function(x + k3_x * d_h, y + k3_y * d_h, z + k3_z * d_h)
        k_x_array = np.array([k1_x, k2_x, k3_x, k4_x])
        k_y_array = np.array([k1_y, k2_y, k3_y, k4_y])
        k_z_array = np.array([k1_z, k2_z, k3_z, k4_z])
        weights = np.array([1/6, 1/3, 1/3, 1/6])
        x += d_h * np.dot(weights, k_x_array)
        y += d_h * np.dot(weights, k_y_array)
        z += d_h * np.dot(weights, k_z_array)
        x_approx.append(x)
        y_approx.append(y)
        z_approx.append(z)

    return steps, x_approx[:-1], y_approx[:-1], z_approx[:-1]

if __name__ == '__main__':

    # Initial conditions
    X_0 = 0.0
    Y_0 = 1.0
    Z_0 = 1.05
    T_0 = 0.0
    DELTA_H = 0.01
    T_END = 10000.0
    _sigma = 10.0
    _rho = 28.0
    _beta = 2.667

    # Get the gradient at given (X, Y, Z) positions
    NUM_STEPS = 15
    x_coord = np.linspace(-5.0, 5.0, NUM_STEPS).astype(np.float128)
    y_coord = np.linspace(-5.0, 5.0, NUM_STEPS).astype(np.float128)
    z_coord = np.linspace(-5.0, 5.0, NUM_STEPS).astype(np.float128)
    X, Y, Z = np.meshgrid(x_coord, y_coord, z_coord)
    grad_x, grad_y, grad_z = lorenz_equations(X, Y, Z, _sigma, _rho, _beta)
    vec_color = np.sqrt(np.square(grad_x) + np.square(grad_y) + np.square(grad_z))

    # Numerically integrate lorenz equations
    rk4_steps, x_t, y_t, z_t = runge_kutta(lambda x, y, z: lorenz_equations(x, y, z, _sigma, _rho, _beta), T_0, X_0, Y_0, Z_0, DELTA_H, T_END)

    # Plot in phase space
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1, 1, 1, projection='3d')
    axs.set_xlabel('x(t)')
    axs.set_ylabel('y(t)')
    axs.set_zlabel('z(t)')
    axs.set_title('Lorenz system')
    # vectors = axs.quiver(X, Y, Z, grad_x, grad_y, grad_z, normalize=True)
    [line] = axs.plot(x_t, y_t, z_t, c='darkblue', linewidth=0.5)

    # Show plot
    plt.show()