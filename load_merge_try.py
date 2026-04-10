# ==================== 1. DATA LOADING & MERGE ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

fatalities = pd.read_csv('fatalities.csv', low_memory=False)
crashes = pd.read_csv('crashes.csv', low_memory=False)

df = pd.merge(fatalities, crashes, on='Crash ID', how='left')
print("Merged shape:", df.shape)

# ==================== 2. CLEANING ====================
df.replace(-9, np.nan, inplace=True)

# === CRITICAL FIX: Clean Speed Limit after merge ===
print("Cleaning Speed Limit column...")
df['Speed Limit_x'] = df['Speed Limit_x'].replace('<40', 35)          # reasonable value
df['Speed Limit_x'] = df['Speed Limit_x'].replace('Unspecified', np.nan)
df['Speed Limit_x'] = pd.to_numeric(df['Speed Limit_x'], errors='coerce')

df['Speed Limit_y'] = df['Speed Limit_y'].replace('<40', 35)
df['Speed Limit_y'] = df['Speed Limit_y'].replace('Unspecified', np.nan)
df['Speed Limit_y'] = pd.to_numeric(df['Speed Limit_y'], errors='coerce')

# Median imputation
num_cols = ['Age', 'Speed Limit_x', 'Speed Limit_y', 'Number Fatalities']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Mode imputation for categorical
cat_cols = ['State_x', 'State_y', 'Crash Type_x', 'Crash Type_y', 'Road User',
            'National Remoteness Areas_x', 'National Remoteness Areas_y',
            'Dayweek', 'Time of day', 'Christmas Period', 'Easter Period']

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

print(f"Shape after cleaning: {df.shape}")
print(f"Missing values left: {df.isnull().sum().sum()}\n")

# ==================== 3. FEATURE ENGINEERING + STANDARDISATION ====================
cat_features = [col for col in ['State_x', 'State_y', 'Crash Type_x', 'Crash Type_y',
                                'Road User', 'National Remoteness Areas_x',
                                'National Remoteness Areas_y', 'Dayweek',
                                'Time of day', 'Christmas Period', 'Easter Period']
                if col in df.columns]

X_cat = pd.get_dummies(df[cat_features], drop_first=True)

num_features = ['Speed Limit_x', 'Speed Limit_y', 'Year', 'Month']
X_num = df[[col for col in num_features if col in df.columns]]

X = pd.concat([X_cat, X_num], axis=1)

y_age = df['Age']

# Standardisation
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std = pd.DataFrame(X_std, columns=X.columns)

print(f"Final feature matrix shape: {X_std.shape}")
print("Standardisation completed successfully!")