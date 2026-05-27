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

# ===================================================================================================================================================

def diffusion_path(t: float=10.0):
    
    time_steps = np.linspace(0, t, num=1000)
    brownian_motion = np.random.normal(loc=0.0, scale=np.sqrt(time_steps[1] - time_steps[0]), size=time_steps.shape)
    diffusion_values = np.cumsum(brownian_motion)

    return time_steps, diffusion_values

# ===================================================================================================================================================

def plot(T: float):

    _, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Left plot: 50 Diffusion paths
    for _ in range(50):
        t_values, diffusion_values = diffusion_path(t=T)
        axes[0].plot(t_values, diffusion_values, alpha=0.3, color='tab:blue', linewidth=1)
    
    axes[0].set_title("50 Realizations of Brownian Motion")
    axes[0].set_xlabel("Time ($t$)")
    axes[0].set_ylabel("Value ($X_t$)")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    for _ in range(50):
        t_values, kac_values = kac_path(t=T, wave_front_speed=4.0, change_rate=2.0)
        axes[1].plot(t_values, kac_values, alpha=0.3, color='tab:green', linewidth=1)
    
    axes[1].set_title("50 Kac Realizations ($c=4$, $a=2$)")
    axes[1].set_xlabel("Time ($t$)")
    axes[1].set_ylabel("Value of Kac Process ($X_t$)")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    for _ in range(50):
        t_values, kac_values = kac_path(t=T, wave_front_speed=100.0, change_rate=10.0)
        axes[2].plot(t_values, kac_values, alpha=0.3, color='tab:red', linewidth=1)
    
    axes[2].set_title("50 Kac Realizations ($c=100$, $a=10$)")
    axes[2].set_xlabel("Time ($t$)")
    axes[2].set_ylabel("Value of Kac Process ($X_t$)")
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("kac_diffusion_comparison.png", dpi=300, bbox_inches='tight')

# ===================================================================================================================================================

if __name__ == "__main__":
    plot(T=10.0)