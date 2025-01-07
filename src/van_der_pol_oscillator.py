import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

# Define global variables
T_END = 25.0

def runge_kutta(function, t_0, x_0, y_0, d_h, t_end=T_END):
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
        y += d_h * np.dot(weights, k_y_array)
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

    F_x = (1/3) * np.pow(x,3) - x
    x_prime = mu * (y - F_x)
    y_prime = -(1/mu) * x
    return x_prime, y_prime


if __name__ == '__main__':

    # Initial conditions
    X_0 = 0.0
    Y_0 = 0.0
    T_0 = 0.0
    DELTA_H = 0.005
    _mu_0 = 20.0

    # Now try that with the Van der Pol oscillator
    rk4_vdp_steps, x_t_vdp, y_t_vdp = runge_kutta(lambda x, y: vdp_prime(x, y, mu=_mu_0), T_0, X_0, Y_0, DELTA_H, t_end=25)

    # Get the gradient at given (X, Y) positions
    NUM_STEPS = 30
    x_coord = np.linspace(-4.0, 4.0, NUM_STEPS).astype(np.float128)
    y_coord = np.linspace(-4.0, 4.0, NUM_STEPS).astype(np.float128)

    X, Y = np.meshgrid(x_coord, y_coord)
    grad_x, grad_y = vdp_prime(X, Y, mu=_mu_0)
    vec_color = np.sqrt(np.square(grad_x) + np.square(grad_y))

    # Plot dynamics in phase space
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.25, bottom=0.25)    # Adjust the subplots region to leave some space for the sliders and buttons
    [line] = axs.plot(x_t_vdp, y_t_vdp, c='darkblue', linewidth=0.5)  # 'line' variable is used for modifying the line later
    vectors = axs.quiver(X, Y, grad_x, grad_y, vec_color, scale_units='xy')    # 'vectors' variable can be accessed and modified later
    axs.set_xlabel('x(t)')
    axs.set_ylabel('y(t)')
    axs.set_title('Van der Pol oscillator - Interactive')
    axs.set_xlim([-4.0, 4.0])
    axs.set_ylim([-4.0, 4.0])
    axs.grid()
    
    ## Def everything necessray for sliders and buttons
    # Slider for dampening parameter mu
    mu_slider_ax = fig.add_axes([0.2, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    mu_slider = Slider(mu_slider_ax, 'mu', valmin=0.1, valmax=50.0, valinit=_mu_0)
    
    # Slider for x-coordinate
    x_slider_ax = fig.add_axes([0.2, 0.125, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    x_slider = Slider(x_slider_ax, 'x', valmin=-4.0, valmax=4.0, valinit=X_0)
    
    # Slider for y-cooordinate
    y_slider_ax = fig.add_axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    y_slider = Slider(y_slider_ax, 'y', valmin=-4.0, valmax=4.0, valinit=Y_0)
    
    # Reset button
    reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_button_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

    # Show/Hide button
    sh_button_ax = fig.add_axes([0.025, 0.5, 0.1, 0.1], facecolor='lightgoldenrodyellow')
    sh_button = CheckButtons(sh_button_ax, ('Graph', 'Vectors'), actives=(True, True))  # Make both 'Graph' and 'Vectors' visible initially.

    # Define an action for modifying the line when any slider's value changes
    def slider_on_changed(val):
        """Changes the dampening parameter mu, x(0) and/or y(0) and reevaluates the function at that point."""
        
        # Evaluate differential equation with new parameter
        _, x_t_vdp, y_t_vdp = runge_kutta(lambda x, y: vdp_prime(x=x, y=y, mu=mu_slider.val), t_0=T_0, x_0=x_slider.val, y_0=y_slider.val, d_h=DELTA_H, t_end=25)
        
        # Evaluate vector field with new parameter
        grad_x, grad_y = vdp_prime(X, Y, mu=mu_slider.val)

        # Update graph and vector field
        line.set_ydata(y_t_vdp)
        line.set_xdata(x_t_vdp)
        vectors.set_UVC(U=grad_x, V=grad_y)
        fig.canvas.draw_idle()
    
    # Add a button for resetting the parameters
    def reset_button_on_clicked(mouse_event):
        """Resets the parameters to initial values."""
        mu_slider.reset()
        x_slider.reset()
        y_slider.reset()
    
    # Add button for showing/hiding graph
    def set_visible_on_clicked(label):
        """Shows/hides graph when checked/unchecked."""
        if label == 'Graph':
            line.set_visible(not line.get_visible())
            plt.draw()
        else:
            vectors.set_visible(not vectors.get_visible())
            plt.draw()

    # Manage actions
    mu_slider.on_changed(slider_on_changed)
    x_slider.on_changed(slider_on_changed)
    y_slider.on_changed(slider_on_changed)
    reset_button.on_clicked(reset_button_on_clicked)
    sh_button.on_clicked(set_visible_on_clicked)
        
    ## In a second plot, show superimposed dynamics
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1,1,1)
    axs.set_title("Van der Pol oscillator - Phase")
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.set_xlim([-4, 4])
    axs.set_ylim([-4, 4])
    axs.grid()

    NUM_STEPS = 5
    x_coord = np.linspace(-4.0, 4.0, NUM_STEPS).astype(np.float128)
    y_coord = np.linspace(-4.0, 4.0, NUM_STEPS).astype(np.float128)

    # Iterate over all x coordinates
    for i in x_coord:

        # Iterate over all y coordinates
        for j in y_coord:

            # Evaluate ODEs at every possible (x_0, y_0)
            rk4_vdp_steps, x_t_vdp, y_t_vdp = runge_kutta(lambda x, y: vdp_prime(x, y, mu=_mu_0), T_0, i, j, DELTA_H, t_end=25)

            # Plot the dynamics
            axs.plot(x_t_vdp, y_t_vdp, c='k', linewidth=0.2, alpha=0.65)
    
    ## Visualize vectorfield
    # Get the gradient at given (X, Y) positions
    NUM_STEPS = 30
    x_coord = np.linspace(-4.0, 4.0, NUM_STEPS).astype(np.float128)
    y_coord = np.linspace(-4.0, 4.0, NUM_STEPS).astype(np.float128)

    X, Y = np.meshgrid(x_coord, y_coord)
    grad_x, grad_y = vdp_prime(X, Y, mu=_mu_0)

    ## In a third plot, show superimposed dynamics
    fig = plt.figure(figsize=(10, 10))
    axs = fig.add_subplot(1,1,1)
    axs.set_title("Van der Pol oscillator - Vectorfield")
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.set_xlim([-4, 4])
    axs.set_ylim([-4, 4])
    axs.grid()
    axs.quiver(X, Y, grad_x, grad_y)

    # Show all plots
    plt.show()
