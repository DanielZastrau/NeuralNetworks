import numpy as np
import matplotlib.pyplot as plt

def plot_kac_paths(num_paths: int = 200):

    plt.figure(figsize=(10, 6))

    b = 1
    num_steps = 1000
    T = 5

    t_vals = np.linspace(0, T, num_steps)


    # U = np.random.uniform(-1, 1, num_paths)
    U = np.linspace(-1, 1, num_paths)

    paths = np.zeros(shape=(num_paths, num_steps))

    for i, t_val in enumerate(t_vals):
        width_of_the_reached_dist = b * (1 - np.exp(-t_val / b))

        paths[:, i] = U * width_of_the_reached_dist

    for path in paths:
        plt.plot(t_vals, path)

    plt.title(f"{num_paths} Realizations of a MMD Process")
    plt.xlabel("Time ($t$)")
    plt.ylabel("Value of MMD Process ($X_t$)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    plot_kac_paths(num_paths=20)