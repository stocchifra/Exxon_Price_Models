import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Download historical data for Exxon Mobil Corporation (XOM) from Yahoo Finance
ticker = 'XOM'
data = yf.download(ticker, start="2010-01-01", end="2023-11-16")

# Calculate daily returns
data['Returns'] = data['Adj Close'].pct_change()

# Drop any NaNs
data.dropna(inplace=True)

# Get the mean and standard deviation of daily returns
mean_daily_return = data['Returns'].mean()
std_dev_return = data['Returns'].std()

# Set the number of simulations and the time horizon (days until 1 March 2024)
num_simulations = 100000
days_to_predict = np.busday_count('2023-11-16', '2024-03-01')

# Create an empty matrix to hold the simulation results
simulation_results = np.zeros((num_simulations, days_to_predict))

# Run the simulations
np.random.seed(42)  # For reproducibility
for sim in range(num_simulations):
    # Start with the last known price
    current_price = data['Adj Close'][-1]
    for d in range(days_to_predict):
        # Simulate the price change
        price_change = current_price * (1 + np.random.normal(mean_daily_return, std_dev_return))
        simulation_results[sim, d] = price_change
        current_price = price_change

# Convert the results to a DataFrame for easier analysis
simulation_df = pd.DataFrame(simulation_results)

# Plot the simulations
plt.figure(figsize=(10, 6))
plt.plot(simulation_df.T, color='blue', alpha=0.1)
plt.title('Monte Carlo Simulation of XOM Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

# Show statistical summary on the final day
final_day_stats = simulation_df.iloc[:, -1].describe()
print(final_day_stats)
