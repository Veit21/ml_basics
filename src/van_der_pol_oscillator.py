import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# Define global variables
T_END = 25

def runge_kutta(function, t_0, x_0, y_0, d_h, t_end=T_END):
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

def vdp_prime(x, y, mu=2.0):
    """Van der Pol oscillator. System of two interacting equations.

    Args:
        x (float): Current x position.
        y (float): Current y position.
        mu (int, optional): Damping parameter. Defaults to 2.

    Returns:
        tuple: Rates of change (dx/dt, dy/dt).
    """

    F_x = (1/3) * x**3 - x
    x_prime = mu * (y - F_x)
    y_prime = -(1/mu)*x
    return x_prime, y_prime

def F_x_function(x):
    """Function of the y nullcline of the Van der Pol oscillator.

    Args:
        x (numpy.ndarray): Input values to the function f(x).

    Returns:
        numpy.ndarray: Function values evaluated at x.
    """
    
    return (1/3) * x**3 - x

if __name__ == '__main__':

    # Initial conditions
    X_0 = 0.0
    Y_0 = 0.0
    T_0 = 0
    DELTA_H = 0.01
    _mu_0 = 20.0

    # Now try that with the Van der Pol oscillator
    rk4_vdp_steps, x_t_vdp, y_t_vdp = runge_kutta(lambda x, y: vdp_prime(x, y, mu=_mu_0), T_0, X_0, Y_0, DELTA_H, t_end=25)
    """
    # Initial conditions
    X_0 = 8.0
    Y_0 = 5.0

    # Now try that with the Van der Pol oscillator
    rk4_vdp_steps_2, x_t_vdp_2, y_t_vdp_2 = runge_kutta(lambda x, y: vdp_prime(x, y, mu=_mu_0), T_0, X_0, Y_0, DELTA_H, t_end=25)

    # Initial conditions
    X_0 = -7.0
    Y_0 = -4.0

    # Now try that with the Van der Pol oscillator
    rk4_vdp_steps_3, x_t_vdp_3, y_t_vdp_3 = runge_kutta(lambda x, y: vdp_prime(x, y, mu=_mu_0), T_0, X_0, Y_0, DELTA_H, t_end=25)"""

    # Get the y nullcline values
    x_nullcline = np.linspace(-7.0, 8.0, 100)
    y_nullcline = F_x_function(x_nullcline)

    # Plot dynamics in phase space
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom=0.25)    # Adjust the subplots region to leave some space for the sliders and buttons
    [line] = axs.plot(x_t_vdp, y_t_vdp, c='k', linewidth=0.75)  # 'line' variable is used for modifying the line later
    # axs.plot(x_t_vdp_2, y_t_vdp_2, c='c', linewidth=0.75)
    # axs.plot(x_t_vdp_3, y_t_vdp_3, c='r', linewidth=0.75)
    # axs.scatter(x_nullcline, y_nullcline, c='k', s=4)   # Plot y nullcline
    axs.set_xlabel('x(t)')
    axs.set_ylabel('y(t)')
    axs.set_title('Van der Pol oscillator')
    axs.set_xlim([-10, 10])
    axs.set_ylim([-10, 10])
    axs.grid()
    
    ## Def everything necessray for sliders and buttons
    # Slider for dampening parameter mu
    mu_slider_ax = fig.add_axes([0.2, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    mu_slider = Slider(mu_slider_ax, 'mu', valmin=0.1, valmax=50.0, valinit=_mu_0)
    
    # Slider for x-coordinate
    x_slider_ax = fig.add_axes([0.2, 0.125, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    x_slider = Slider(x_slider_ax, 'x', valmin=-10.0, valmax=10.0, valinit=X_0)
    
    # Slider for y-cooordinate
    y_slider_ax = fig.add_axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    y_slider = Slider(y_slider_ax, 'y', valmin=-10.0, valmax=10.0, valinit=Y_0)
    
    # Reset button
    reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_button_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

    # Define an action for modifying the line when any slider's value changes
    def slider_on_changed(val):
        """Changes the dampening parameter mu, x(0) and/or y(0) and reevaluates the function at that point."""
        
        # Evaluate differential equation with new parameter
        _, x_t_vdp, y_t_vdp = runge_kutta(lambda x, y: vdp_prime(x=x_slider.val, y=y_slider.val, mu=mu_slider.val), T_0, X_0, Y_0, DELTA_H, t_end=25)
        
        # Set the new y-values
        line.set_ydata(y_t_vdp)
        line.set_xdata(x_t_vdp)
        fig.canvas.draw_idle()
    
    # Add a button for resetting the parameters
    def reset_button_on_clicked(mouse_event):
        """Resets the parameters to initial values."""
        mu_slider.reset()
        x_slider.reset()
        y_slider.reset()
    
    # Manage actions
    mu_slider.on_changed(slider_on_changed)
    x_slider.on_changed(slider_on_changed)
    y_slider.on_changed(slider_on_changed)
    reset_button.on_clicked(reset_button_on_clicked)
        
    plt.show()
