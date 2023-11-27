import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import datetime

# Fetch historical stock data of XOM from Yahoo Finance
ticker = "XOM"
start_date = "2010-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(ticker, start=start_date, end=end_date)

# Prepare the data for Prophet
df_prophet = pd.DataFrame({
    'ds': data.index,
    'y': data['Close']
})

# Create and fit the Prophet model
model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
model.fit(df_prophet)

# Create a DataFrame for future dates
future_dates = model.make_future_dataframe(periods=365)  # Extend a bit beyond the target date

# Make predictions
forecast = model.predict(future_dates)

# Extract the prediction for March 1, 2024
forecasted_date = pd.to_datetime("2024-03-01")
prediction = forecast[forecast['ds'] == forecasted_date]['yhat'].values[0]

print(f"Predicted stock price for XOM on March 1, 2024: ${prediction:.2f}")

# Optional: Plotting the forecast
from matplotlib import pyplot as plt

fig1 = model.plot(forecast)
plt.show()

# Optional: Plotting forecast components
fig2 = model.plot_components(forecast)
plt.show()
