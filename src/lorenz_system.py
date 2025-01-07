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
    y_dot = x * rho - x * z - y
    z_dot = x * y - beta * z

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

    # Iterate over time steps. Simulation loop.
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

def forward_euler(function, t_0, x_0, y_0, z_0, d_h, t_end):
    """Forward-Euler method to numerically integrate (a system of) ODEs

    Args:
        function (function): function to be integrated.
        t_0 (flaot): Initial t.
        x_0 (float): Initial x.
        y_0 (float): Initial y.
        z_0 (float): Initial z.
        d_h (float): Step width.
        t_end (float): Final step of simulation.

    Returns:
        tuple: ...
    """

    # Initial conditions
    x = x_0
    y = y_0
    z = z_0

    # Simulate over discrete time steps
    steps = np.arange(t_0, t_end + d_h, d_h)
    
    # Keep track of the (x,y) positions
    x_approx = [x]
    y_approx = [y]
    z_approx = [z]

    # Iterate over time steps. Simulation loop.
    for t in steps:
        x_dot, y_dot, z_dot = function(x, y, z)
        x += d_h * x_dot
        y += d_h * y_dot
        z += d_h * z_dot
        x_approx.append(x)
        y_approx.append(y)
        z_approx.append(z)

    return steps, x_approx[:-1], y_approx[:-1], z_approx[:-1]

if __name__ == '__main__':

    # Initial conditions
    X_0 = 1.0
    Y_0 = 1.0
    Z_0 = 1.0
    T_0 = 0.0
    DELTA_H = 0.01
    T_END = 100.0
    _sigma_0 = 10.0
    _rho_0 = 28.0
    _beta_0 = 2.66

    # Get the gradient at given (X, Y, Z) positions
    NUM_STEPS = 5
    x_coord = np.linspace(-5.0, 5.0, NUM_STEPS).astype(np.float128)
    y_coord = np.linspace(-5.0, 5.0, NUM_STEPS).astype(np.float128)
    z_coord = np.linspace(-5.0, 5.0, NUM_STEPS).astype(np.float128)
    X, Y, Z = np.meshgrid(x_coord, y_coord, z_coord)
    grad_x, grad_y, grad_z = lorenz_equations(X, Y, Z, _sigma_0, _rho_0, _beta_0)
    vec_color = np.sqrt(np.square(grad_x) + np.square(grad_y) + np.square(grad_z))

    # Numerically integrate lorenz equations
    rk4_steps, x_t, y_t, z_t = runge_kutta(lambda x, y, z: lorenz_equations(x, y, z, _sigma_0, _rho_0, _beta_0), T_0, X_0, Y_0, Z_0, DELTA_H, T_END)

    # Plot in phase space
    fig = plt.figure(figsize=(15, 15))
    axs = fig.add_subplot(1, 1, 1, projection='3d')
    fig.subplots_adjust(bottom=0.25)    # Adjust the subplots region to leave some space for the sliders and buttons
    axs.set_xlabel('x(t)')
    axs.set_ylabel('y(t)')
    axs.set_zlabel('z(t)')
    axs.set_title('Lorenz attractor')
    # vectors = axs.quiver(X, Y, Z, grad_x, grad_y, grad_z, normalize=True)
    [line] = axs.plot(x_t, y_t, z_t, c='darkblue', linewidth=0.2)

    ## Def everything necessray for sliders and buttons
    # Slider for sigma parameter
    sigma_slider_ax = fig.add_axes([0.2, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    sigma_slider = Slider(sigma_slider_ax, 'sigma', valmin=0.1, valmax=30.0, valinit=_sigma_0)
    
    # Slider for rho parameter
    rho_slider_ax = fig.add_axes([0.2, 0.125, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    rho_slider = Slider(rho_slider_ax, 'rho', valmin=0.1, valmax=30.0, valinit=_rho_0)
    
    # Slider for beta parameter
    beta_slider_ax = fig.add_axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    beta_slider = Slider(beta_slider_ax, 'beta', valmin=0.1, valmax=30.0, valinit=_beta_0)

    # Reset button
    reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_button_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

    # Define an action for modifying the line when any slider's value changes
    def slider_on_changed(val):
        """Changes the parameters and reevaluates the function at that point."""
        
        # Evaluate differential equation with new parameter
        _, x_t, y_t, z_t = runge_kutta(lambda x, y, z: lorenz_equations(x, y, z,
                                                                        sigma=sigma_slider.val,
                                                                        rho=rho_slider.val,
                                                                        beta=beta_slider.val), T_0, X_0, Y_0, Z_0, DELTA_H, T_END)

        # Update graph and vector field
        line.set_data_3d(x_t, y_t, z_t)
        fig.canvas.draw_idle()
    
    # Add a button for resetting the parameters
    def reset_button_on_clicked(mouse_event):
        """Resets the parameters to initial values."""
        sigma_slider.reset()
        rho_slider.reset()
        beta_slider.reset()

    # Manage actions
    sigma_slider.on_changed(slider_on_changed)
    rho_slider.on_changed(slider_on_changed)
    beta_slider.on_changed(slider_on_changed)
    reset_button.on_clicked(reset_button_on_clicked)

    ## Test the initialization sensitivity
    # Set x_0 to 1 + epsilon. Plot multiple trajectories
    EPSILON = 0.01 # Perturbation parameter
    N = 0  # N runs to compare
    for i in range(N):
        rk4_steps, x_t_hat, y_t_hat, z_t_hat = runge_kutta(lambda x, y, z: lorenz_equations(x, y, z, _sigma_0, _rho_0, _beta_0), T_0, (X_0 + i*EPSILON), Y_0, Z_0, DELTA_H, T_END)
        axs.plot(x_t_hat, y_t_hat, z_t_hat, linewidth=0.2)

    ## Test bifurcations for different rho
    # Plot bifurcation plot
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1, 1, 1, projection='3d')
    axs.set_xlabel('x(t)')
    axs.set_ylabel('y(t)')
    axs.set_zlabel('z(t)')
    axs.set_title('Lorenz system - Bifurcation plot')

    RHOS = np.linspace(1., 25., 50)
    for a_rho in RHOS:
        rk4_steps, x_t, y_t, z_t = runge_kutta(lambda x, y, z: lorenz_equations(x, y, z, _sigma_0, a_rho, _beta_0), T_0, X_0, Y_0, Z_0, DELTA_H, T_END)
        axs.scatter(x_t[-1], y_t[-1], z_t[-1], c='darkblue', s=4)
    
    # Show plot
    plt.show()
