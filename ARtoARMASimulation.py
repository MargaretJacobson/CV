import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# --- Define Target and Calculate Simulation Length ---
target_num_agg_obs = 1000000
m = 1 # Aggregation factor
burnin = 300
seed = 2 # From your previous code

# Calculate required number of post-burnin daily observations
obs_needed = target_num_agg_obs * m

# Calculate total simulation length required
sim = obs_needed + burnin

# Calculate actual number of post-burnin obs based on sim and burnin
# (This should match obs_needed)
obs = sim - burnin

print(f"Target aggregated observations: {target_num_agg_obs}")
print(f"Aggregation factor (m): {m}")
print(f"Required post-burnin daily obs (obs_needed): {obs_needed}")
print(f"Burn-in period: {burnin}")
print(f"Total simulation length (sim): {sim}")
print(f"Actual post-burnin daily obs (obs): {obs}")
# --- End of Calculation ---


# Model Parameters
sigma_eps = 0.01 # This is a standard deviation
rho = 0.99
phi = 1.05

# Set random seed for reproducibility
np.random.seed(seed)

# %% Generate Shocks and AR(1) Process x_t
eps = stats.norm.rvs(loc=0, scale=sigma_eps, size=sim) #Note: Scale is standard deviation. 

# Initialize x array (using zeros is standard, MATLAB started at x(1)=1)
x = np.zeros(sim)
x[0] = 1 # Replicating MATLAB's starting value

for t in range(1, sim):
    x[t] = rho * x[t-1] + eps[t]

# %% Calculate Equilibrium Inflation pi_t (post burn-in)
# Make sure phi is not equal to rho to avoid division by zero
if phi == rho:
    raise ValueError("phi cannot be equal to rho")

# Select x after burn-in period
x_effective = x[burnin:]

# Calculate pi
pi = -x_effective / (phi - rho)

# %% Aggregate Inflation pi_t to AggPi_T
num_agg_obs = obs // m # Number of full aggregated periods
print(f"Number of daily observations (post-burnin): {obs}")
print(f"Number of aggregated observations: {num_agg_obs}")

# Ensure we only use full blocks of size m
pi_trunc = pi[:num_agg_obs * m]

# Reshape and calculate the mean for each block
# Reshape into (num_agg_obs, m), then take mean over axis 1 (columns)
AggPi = np.mean(pi_trunc.reshape(num_agg_obs, m), axis=1)

print(f"Shape of pi (daily): {pi.shape}")
print(f"Shape of AggPi (aggregated): {AggPi.shape}")
print("First few aggregated values:", AggPi[:5]) # Optional: check values

# %% Check Simulated / Aggregated Data (Manual check equivalent)
#AggPi1 = (1/m) * np.sum(pi[0:m])
#AggPi2 = (1/m) * np.sum(pi[m:2*m])
#AggPi3 = (1/m) * np.sum(pi[2*m:3*m])
#print("Manual Agg Check (first 3):", AggPi1, AggPi2, AggPi3)


# Estimate AR(1) on disaggregated Data and ARMA(1,1) on aggregated data.
print("\nEstimating models...")
# Estimate AR(1) for pi
# Note: trend='n' forces intercept/constant to zero
#model_ar1 = ARIMA(pi, order=(1, 0, 0), trend='n')
#results_ar1 = model_ar1.fit()
#print("\n--- AR(1) Estimation Results (Disaggregated pi_t) ---")
#print(results_ar1.summary())
#rho_hat_ar1 = results_ar1.arparams[0]
#sigma2_w_hat = np.var(results_ar1.resid) # Variance of residuals w_t
#print(f"Estimated rho: {rho_hat_ar1:.4f}") # should be rho =0.99
#print(f"Estimated sigma^2_w: {sigma2_w_hat:.6f}") # should be (sigma^2/(phi-rho)^2) =0.0001/(1.05-0.99)^2 = 0.0277


# Estimate ARMA(1,1) for AggPi
model_arma11 = ARIMA(AggPi, order=(1, 0, 1), trend='n')
results_arma11 = model_arma11.fit()
print("\n--- ARMA(1,1) Estimation Results (Aggregated AggPi_T) ---")
print(results_arma11.summary())
rho_m_hat = results_arma11.arparams[0]
theta_hat = results_arma11.maparams[0]
# CORRECT LINE BELOW - uses results_arma11 and assigns to sigma2_u_hat
sigma2_u_hat = np.var(results_arma11.resid) # Variance of residuals u_t
print(f"Estimated rho^m: {rho_m_hat:.4f}")
print(f"Estimated theta: {theta_hat:.4f}")
print(f"Estimated sigma^2_u: {sigma2_u_hat:.6f}")


# %% Pull Parameters to calculate Variance
print("\nCalculating variances...")

# We need to make sure the estimation ran successfully and gave us results
# A simple check is if the results variables exist. A more robust check
# might involve checking if the fit converged.
calculation_possible = True
try:
    # Check if necessary variables from AR(1) estimation exist
  #  rho_hat_ar1
  #  sigma2_w_hat
    # Check if necessary variables from ARMA(1,1) estimation exist
    rho_m_hat
    theta_hat
    sigma2_u_hat
except NameError:
    print("ERROR: Estimation results variables not found. Cannot calculate variances.")
    calculation_possible = False

if calculation_possible:
    # --- Variance of the underlying AR(1) process pi_t ---
    print("-" * 30) # Separator
    # Check for stationarity: |rho_hat| < 1
   # if abs(rho_hat_ar1) < 1:
        # Formula: Var(pi) = Var(w) / (1 - rho^2)
    #    var_pi_disagg = sigma2_w_hat / (1 - rho_hat_ar1**2)
     #   print(f"Implied Variance of Disaggregated Process (pi_t): {var_pi_disagg:.6f}")
      #  print(f"(Using rho_hat={rho_hat_ar1:.4f}, sigma2_w_hat={sigma2_w_hat:.6f})")
    #else:
     #   print(f"Estimated AR(1) parameter |rho_hat| = {abs(rho_hat_ar1):.4f} >= 1.")
      #  print("Disaggregated variance calculation skipped (non-stationary or unit root).")

    # --- Variance of the aggregated ARMA(1,1) process AggPi_T ---
    print("-" * 30) # Separator
    # Check for stationarity: |rho_m_hat| < 1
    if abs(rho_m_hat) < 1:
        # Formula: Var(AggPi) = sigma_u^2 * (1 + theta^2 + 2*rho^m*theta) / (1 - (rho^m)^2)
        # Using estimated parameters
        numerator = 1 + theta_hat**2 + 2 * rho_m_hat * theta_hat
        denominator = 1 - rho_m_hat**2
        var_agg_arma = sigma2_u_hat * numerator / denominator
        print(f"Implied Variance of Aggregated Process (AggPi_T): {var_agg_arma:.6f}")
        print(f"(Using rho_m_hat={rho_m_hat:.4f}, theta_hat={theta_hat:.4f}, sigma2_u_hat={sigma2_u_hat:.6f})")
    else:
        print(f"Estimated ARMA(1,1) AR parameter |rho_m_hat| = {abs(rho_m_hat):.4f} >= 1.")
        print("Aggregated variance calculation skipped (non-stationary or unit root).")
    print("-" * 30) # Separator


# %% Plot Spectrum of MA Filter
# --- UNCOMMENT BELOW TO RUN PLOTTING ---
# print("\nPlotting MA Filter Spectrum...")
# plt.figure(figsize=(10, 6))
# omega = np.linspace(1e-6, np.pi, 500) # Start slightly away from 0 for numerical stability

# for m_plot in range(2, 31): # Plot for m = 2 to 30
#     # Gain function G(omega) = (1/m^2) * |(1-exp(-i*omega*m))/(1-exp(-i*omega))|^2
#     # Simplified form: G(omega) = (1/m^2) * (1 - cos(m*omega)) / (1 - cos(omega))
#     gain = (1 - np.cos(m_plot * omega)) / (m_plot**2 * (1 - np.cos(omega)))
#     # Handle limit at omega=0 (should be 1/m^2 * m^2/1 = 1, but calculation gives NaN)
#     # The formula as written G = (1-cos)/(m^2(1-cos)) has limit 1.
#     # Let's use the formula from the text: (1/m^2) * (1-cos(mw))/(1-cos(w))
#     # The limit of (1-cos(ax))/(1-cos(bx)) as x->0 is (a/b)^2 using L'Hopital twice.
#     # So limit of (1-cos(m*omega))/(1-cos(omega)) as omega->0 is m^2.
#     # Thus the gain formula limit is (1/m^2) * m^2 = 1 at omega=0.
#     # We started omega slightly > 0, so it should be fine.
#     plt.plot(omega, gain, label=f'm={m_plot}' if m_plot % 5 == 0 or m_plot==2 else None)

# plt.title('Gain Function of Temporal Averaging Filter')
# plt.xlabel('Frequency ($\omega$)')
# plt.ylabel('Gain $G(\omega)$')
# plt.xticks([0, np.pi/2, np.pi], ['0', '$\pi/2$', '$\pi$'])
# plt.xlim([0, np.pi])
# plt.ylim(bottom=0) # Ensure y-axis starts at 0
# # plt.legend() # Uncomment to show legend for labeled lines
# plt.grid(True)
# plt.show()

print("\nScript finished.")