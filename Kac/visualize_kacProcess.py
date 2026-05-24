import numpy as np
import matplotlib.pyplot as plt

def poisson_path(rate: float = 2.0, t: float=10.0):
    
    # Generate inter-arrival times: average count is rate * duration
    # We generate a large enough buffer of inter-arrival times
    max_events = int(rate * t * 5)
    inter_arrivals = np.random.exponential(1/rate, [1, max_events])
    arrival_times = np.cumsum(inter_arrivals, axis=1)
    valid_times = arrival_times[arrival_times <= t]

    # Prepend 0 to properly ground the step plot at origin
    t_values = np.concatenate(([0], valid_times.flatten()))
    n_values = np.arange(len(t_values))

    return t_values, n_values

def kac_path(t: float, wave_front_speed: float = 3.0, change_rate: float = 2.0):

    poisson_step_times, poisson_step_values = poisson_path(rate=change_rate, t=t)

    kac_change_times = poisson_step_times.copy()
    kac_change_values = np.zeros(len(poisson_step_values), dtype=float)

    initial_direction = np.random.choice([-1, 1])

    for i in range(1, len(poisson_step_times)):

        time_diff = poisson_step_times[i] - poisson_step_times[i-1]
        integrand = (-1)**poisson_step_values[i-1]

        kac_change_values[i] = kac_change_values[i-1] + initial_direction * wave_front_speed * integrand * time_diff

    return kac_change_times, kac_change_values

def plot_kac_paths(T: float, wave_front_speed: float = 3.0, change_rate: float = 2.0, num_paths: int = 200):

    plt.figure(figsize=(10, 6))

    for _ in range(num_paths):
        t_values, kac_values = kac_path(t=T, wave_front_speed=wave_front_speed, change_rate=change_rate)

        # Plotting the current path
        plt.plot(t_values, kac_values, alpha=0.3, color='tab:blue', linewidth=1)

    plt.title(f"{num_paths} Realizations of a Kac Process ($c={wave_front_speed}$, $\\lambda={change_rate}$)")
    plt.xlabel("Time ($t$)")
    plt.ylabel("Value of Kac Process ($X_t$)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    plot_kac_paths(T=10.0, wave_front_speed=3.0, change_rate=2.0, num_paths=200)