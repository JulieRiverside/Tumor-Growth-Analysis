import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def tumor_growth(t,N,r,K,d0,alpha,c):
    d = d0 * np.exp(-alpha * t)
    dNdT=r*N*(1-N/K)-d*N-c*N #Growth-Chemotherapy effect - immune response
    return dNdT

#Setting Imaginary Parameters
r = 0.2      # Tumor growth rate
K = 1000     # Carrying capacity
d0 = 0.1     # Initial chemotherapy death rate
alpha = 0.02 # Drug resistance rate
c = 0.05     # Immune response rate
N0 = 50      # Initial tumor size

# Time span
t_span = (0, 100)
t_eval = np.linspace(*t_span, 500)

# Solve the ODE without treatment (d=0) for comparison
sol_no_treatment = solve_ivp(tumor_growth, t_span, [N0], args=(r, K, 0, alpha, c), t_eval=t_eval)

# Solve the ODE with chemotherapy
sol_with_treatment = solve_ivp(tumor_growth, t_span, [N0], args=(r, K, d0, alpha, c), t_eval=t_eval)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(sol_no_treatment.t, sol_no_treatment.y[0], label="No Treatment", color="blue")
plt.plot(sol_with_treatment.t, sol_with_treatment.y[0], label="With Chemotherapy & Immune Response", color="red", linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Tumor Size (N)")
plt.title("Tumor Growth with Immune Response & Drug Resistance")
plt.legend()
plt.grid()
plt.show()

#Taking stochastic effects into consideration irl 

sigma = 10   # Noise intensity (higher = more randomness)

#time span for simulation
dt = 0.1  # Time step
T = 100   # Total time
steps = int(T / dt)
t = np.linspace(0, T, steps)

# Initialize tumor size array
N = np.zeros(steps)
N[0] = N0

# Stochastic Simulation using Euler-Maruyama method
for i in range(1, steps):
    d = d0 * np.exp(-alpha * t[i])  # Drug resistance effect
    noise = sigma * np.random.randn() * np.sqrt(dt)  # Stochastic term
    dN = r * N[i-1] * (1 - N[i-1] / K) - d * N[i-1] - c * N[i-1] + noise
    N[i] = max(N[i-1] + dN * dt, 0)  # Ensure tumor size is non-negative

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(t, N, label="Stochastic Tumor Growth", color="purple")
plt.xlabel("Time")
plt.ylabel("Tumor Size (N)")
plt.title("Tumor Growth with Stochastic Effects (Noise in Growth & Treatment)")
plt.legend()
plt.grid()
plt.show()