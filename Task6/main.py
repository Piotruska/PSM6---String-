import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
L = np.pi  # Length of the string
N = 10  # Number of divisions on the string
dx = L / N  # Spatial step size
T = 10  # Total time for the simulation
dt = 0.005  # Time step size
frame_skip = 10  # Number of frames to skip for faster animation


def initial_conditions(L, N):
    x = np.linspace(0, L, N + 1)  # Spatial points
    y = np.sin(x)  # Initial displacement
    v = np.zeros(N + 1)  # Initial velocity (at rest)
    return x, y, v


def acceleration(y, N, dx):
    a = np.zeros(N + 1)
    a[1:-1] = (y[:-2] - 2 * y[1:-1] + y[2:]) / dx ** 2
    return a


def kinetic_energy(v, dx):
    return 0.5 * dx * np.sum(v ** 2)


def potential_energy(y, dx):
    return 0.5 * dx * np.sum((y[:-1] - y[1:]) ** 2 / dx ** 2)


def rk4_step(y, v, dt, N, dx):
    for _ in range(frame_skip):  # Apply multiple simulation steps per frame update
        k1v = dt * acceleration(y, N, dx)
        k1y = dt * v

        k2v = dt * acceleration(y + 0.5 * k1y, N, dx)
        k2y = dt * (v + 0.5 * k1v)

        k3v = dt * acceleration(y + 0.5 * k2y, N, dx)
        k3y = dt * (v + 0.5 * k2v)

        k4v = dt * acceleration(y + k3y, N, dx)
        k4y = dt * (v + k3v)

        v += (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        y += (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    return y, v


def setup_plot():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    ax1.set_xlim(0, L)
    ax1.set_ylim(-1.5, 1.5)
    line1, = ax1.plot([], [], 'b-', label='String Position')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Displacement')

    ax2.set_xlim(0, T)
    ax2.set_ylim(0, 1)
    line2, = ax2.plot([], [], 'r-', label='Kinetic Energy')
    line3, = ax2.plot([], [], 'g-', label='Potential Energy')
    line4, = ax2.plot([], [], 'k-', label='Total Energy')
    ax2.legend()
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Energy [J]')

    return fig, line1, line2, line3, line4


def init(line1, line2, line3, line4):
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4


def update(frame, x, y, v, N, dx, dt, line1, line2, line3, line4, time_data, ke_data, pe_data, te_data):
    y, v = rk4_step(y, v, dt, N, dx)

    ke = kinetic_energy(v, dx)
    pe = potential_energy(y, dx)
    te = ke + pe

    current_time = frame * dt * frame_skip
    time_data.append(current_time)
    ke_data.append(ke)
    pe_data.append(pe)
    te_data.append(te)

    line1.set_data(x, y)
    line2.set_data(time_data, ke_data)
    line3.set_data(time_data, pe_data)
    line4.set_data(time_data, te_data)

    return line1, line2, line3, line4


# Main function to run the animation
def run_animation():
    x, y, v = initial_conditions(L, N)
    fig, line1, line2, line3, line4 = setup_plot()
    time_data, ke_data, pe_data, te_data = [], [], [], []

    ani = FuncAnimation(fig, update,
                        fargs=(x, y, v, N, dx, dt, line1, line2, line3, line4, time_data, ke_data, pe_data, te_data),
                        frames=np.arange(0, int(T / (dt * frame_skip))),
                        init_func=lambda: init(line1, line2, line3, line4),
                        blit=True, repeat=True, interval=20)  # Reduced interval for quicker animation
    plt.show()


run_animation()
