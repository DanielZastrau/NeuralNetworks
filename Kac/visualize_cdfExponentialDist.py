import numpy as np
import matplotlib.pyplot as plt

import argparse

def plot_pdf_exponential_dist(rate: float = 2.0):
    plt.figure(figsize=(10, 6))
    
    t_vals = np.linspace(0, 10, 1000)
    x_vals = 1 - np.exp(-rate * t_vals)
        
    # Plotting the current path
    plt.plot(t_vals, x_vals)

    plt.title(f"Cumulative distribution function of the exponential distribution with rate  {rate}")
    plt.xlabel("Time ($t$)")
    plt.ylabel("x")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'./cdfExponentialDistRate{rate}.png', dpi=300)

# Execution
parser = argparse.ArgumentParser()

parser.add_argument('--rate', type=float)

args = parser.parse_args()
plot_pdf_exponential_dist(rate = args.rate)