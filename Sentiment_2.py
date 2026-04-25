# =========================================
# REGRESSION ANALYSIS
# =========================================

# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================
# 2. LOAD DATASET
# =========================================
df = pd.read_csv("sentiment_dataset.csv")

df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Define features and target
X = df[['Retweets', 'Hour']]
y = df['Likes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Regression Model Trained Successfully!")

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Importance:\n")
print(coefficients)

# =========================================
# TIME SERIES ANALYSIS
# =========================================

# Convert Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set index
df.set_index('Timestamp', inplace=True)

# Posts over time
df.resample('D')['Sentiment'].count().plot()
plt.title("Posts Over Time")
plt.show()

# Sentiment over time
df.groupby([df.index.date, 'Sentiment']).size().unstack().plot()
plt.title("Sentiment Trend Over Time")
plt.show()

# Identify Peak Day Exactly
peak_day = df.resample('D')['Sentiment'].count().idxmax()
print("Peak activity day:", peak_day)