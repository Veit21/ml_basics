import numpy as np
import matplotlib.pyplot as plt

def ode_system(x, y, beta):
    """Nonlinear system of ODEs that form a limit cycle.

    Args:
        x (float): x coordinate.
        y (float): y coordinate.
        beta (float): Parameter that controls relative dynamic rates in x and y directions

    Returns:
        tuple: x_dot, y_dot. First derivatives.
    """
    x_dot = y
    y_dot = -beta * (x + y**3 - y)
    return x_dot, y_dot

def forward_euler(function, X_0, Y_0, T_0, T_END, D_H):
    
    # Initial conditions
    x = X_0
    y = Y_0

    # Simulate over discrete time steps
    steps = np.arange(T_0, T_END + D_H, D_H)

    # Keep track of values
    x_approx = [x]
    y_approx = [y]

    # Iterate
    for n in steps:
        x_dot, y_dot = function(x, y)
        x += D_H * x_dot
        y += D_H * y_dot
        x_approx.append(x)
        y_approx.append(y)
    return steps, x_approx[:-1], y_approx[:-1]

if __name__ == '__main__':

    # Parameters
    _BETA = [1., 10.]
    X_0, Y_0 = np.array([0.1, 0.1]).astype(np.float128)
    _D_H = [0.01, 0.05, 0.1]

    # Prepare plot
    fig = plt.figure(figsize=(20, 10))
    axs1 = fig.add_subplot(1, 2, 1)
    axs2 = fig.add_subplot(1, 2, 2)
    axs1.set_title(f'Simulation beta={_BETA[0]}')
    axs2.set_title(f'Simulation beta={_BETA[1]}')
    axs1.set_xlabel('x(t)')
    axs2.set_xlabel('x(t)')
    axs1.set_ylabel('y(t)')
    axs2.set_ylabel('y(t)')
    axs1.grid()
    axs2.grid()

    # Iterate over parameter combinations
    for beta, axis in zip(_BETA, [axs1, axs2]):
        for d_h in _D_H:

            # Simulate
            _, x_t, y_t = forward_euler(lambda x, y: ode_system(x, y, beta), X_0, Y_0, 0.0, 100, d_h)

            # Plot
            axis.plot(x_t, y_t, '-o', markersize=1, linewidth=0.5, label=f'd_h={d_h:.2f}')
    
    # Show + legend
    axs1.legend(loc='upper right')
    axs2.legend(loc='upper right')
    plt.show()