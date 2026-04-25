# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 2. LOAD DATASET
# =========================================
df = pd.read_csv("3) Sentiment dataset.csv")

print("Dataset Loaded Successfully!\n")

print(df.head())
print(df.columns)

# =========================================
# 3. DATA CLEANING
# =========================================
# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], errors='ignore')


# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Drop missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# check data types
print(df.dtypes)

# Convert Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

print("\nData Cleaning Completed!")

print(df.describe())

# =========================================
# CORRELATION ANALYSIS
# =========================================

# Select numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64'])

# Correlation matrix
corr_matrix = num_cols.corr()

print("\nCorrelation Matrix:\n")
print(corr_matrix)

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()