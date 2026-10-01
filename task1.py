import pandas as pd

# Import the Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Check dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check data types
print("\nData Types:")
print(df.dtypes)
# Handle missing values

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Cabin"] = df["Cabin"].fillna("Unknown")

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# Step 3: Convert categorical features into numerical

# Encode Sex
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# One-hot encode Embarked
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

print("\nDataset After Encoding:")
print(df.head())

print("\nData Types After Encoding:")
print(df.dtypes)
# Step 4: Standardize numerical features

from sklearn.preprocessing import StandardScaler

numerical_columns = ["Age", "Fare", "SibSp", "Parch"]

scaler = StandardScaler()

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\nDataset After Standardization:")
print(df[numerical_columns].head())
# Step 5: Visualize outliers using boxplot

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))

sns.boxplot(data=df[numerical_columns])

plt.title("Boxplot of Numerical Features")
plt.xlabel("Features")
plt.ylabel("Standardized Values")

plt.show()
# Remove outliers using IQR method

Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df[
    ~((df[numerical_columns] < lower_bound) |
      (df[numerical_columns] > upper_bound)).any(axis=1)
]

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nDataset Shape After Removing Outliers:")
print(df_cleaned.shape)
# Save the cleaned dataset
df_cleaned.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")