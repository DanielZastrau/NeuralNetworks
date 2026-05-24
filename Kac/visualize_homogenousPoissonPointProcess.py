import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_poisson_paths(num_paths=200, rate=2.0, duration=10.0):
    plt.figure(figsize=(10, 6))
    
    # Generate inter-arrival times: average count is rate * duration
    # We generate a large enough buffer of inter-arrival times
    max_events = int(rate * duration * 5)
    inter_arrivals = np.random.exponential(1/rate, [num_paths, max_events])
    arrival_times = np.cumsum(inter_arrivals, axis=1)


    # Iterate to filter and plot each path individually
    for i in range(num_paths):
        # Isolate the current path
        path_times = arrival_times[i]
        
        # Filter events that occur within the duration
        valid_times = path_times[path_times <= duration]
        
        # Prepend t=0 to properly ground the step plot at origin
        t_plot = np.concatenate(([0], valid_times))
        n_values = np.arange(len(t_plot))
        
        # Plotting the current path
        plt.step(t_plot, n_values, where='post', alpha=0.3, 
                 color='tab:blue', linewidth=1)

    plt.title(f"{num_paths} Realizations of a Homogeneous Poisson Process ($a={rate}$)")
    plt.xlabel("Time ($t$)")
    plt.ylabel("Number of events ($N_t$)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Execution
plot_multiple_poisson_paths()