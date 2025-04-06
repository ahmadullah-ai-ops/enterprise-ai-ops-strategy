import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dummy data for system usage over time
time = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
cpu_usage = np.array([60, 65, 70, 75, 80])

# Fit a linear regression model to predict future CPU usage
model = LinearRegression()
model.fit(time, cpu_usage)

# Predict usage for the next 2 time periods
future_time = np.array([6, 7]).reshape(-1, 1)
predicted_usage = model.predict(future_time)

print("Predicted CPU usage for the next periods:", predicted_usage)

# Plot the result
plt.plot(time, cpu_usage, label='CPU Usage')
plt.plot(future_time, predicted_usage, label='Predicted Usage', linestyle='--')
plt.legend()
plt.show()

