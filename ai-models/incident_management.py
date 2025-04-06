```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Dummy dataset of incidents (replace with actual monitoring data)
data = {
    'cpu_usage': [55, 80, 45, 90, 72],
    'memory_usage': [60, 85, 40, 95, 80],
    'incident': [0, 1, 0, 1, 1]  # 1 = incident, 0 = no incident
}

df = pd.DataFrame(data)

# Train a simple Random Forest model
X = df[['cpu_usage', 'memory_usage']]
y = df['incident']
model = RandomForestClassifier()
model.fit(X, y)

# Predict if a new set of metrics will trigger an incident
new_metrics = [[85, 75]]  # Example: high CPU and memory usage
prediction = model.predict(new_metrics)

print("Incident Predicted" if prediction[0] == 1 else "No Incident")
