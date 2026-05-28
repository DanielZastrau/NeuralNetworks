import numpy as np
import matplotlib.pyplot as plt

import argparse

def plot_pdf_exponential_dist(rate: float = 2.0):
    plt.figure(figsize=(10, 6))
    
    t_vals_pdf = np.linspace(0, 10, 1000)
    x_vals_pdf = rate * np.exp(-rate * t_vals_pdf)

    # # alternative parameterization of the exponential distribution
    # beta = 1 / rate
    # x_vals = (1 / beta) * np.exp(- t_vals / beta) 

    # plt.plot(t_vals, x_vals, color='red')

    # passing rate to np.random.exponential sets beta so in the lambda def it is equal to 1/beta
    beta = 1 / rate
    t_vals_sampled = np.random.exponential(scale=beta, size=1000)
    x_vals_sampled = np.zeros(dtype=int, shape=1000)

    # 3. Plotting
    plt.figure(figsize=(10, 6))

    # Plot the PDF line
    plt.plot(t_vals_pdf, x_vals_pdf, label=f'PDF ($\lambda={rate}$)', color='red', linewidth=1, alpha=0.5)

    # Plot sampled points
    # We plot them at y=0 (a "rug plot") with transparency (alpha)
    plt.scatter(t_vals_sampled, x_vals_sampled, label='Sampled points', s=5)

    plt.title('Exponential Distribution: PDF and Sampled Points')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'./pdfExponentialDistRate{rate}.png', dpi=300)

# Execution
parser = argparse.ArgumentParser()

parser.add_argument('--rate', type=float)

args = parser.parse_args()
plot_pdf_exponential_dist(rate = args.rate)