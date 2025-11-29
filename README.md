import pandas as pd

df = pd.read_csv(r"C:\Users\O'Bola\loan_data.csv")

df

df.head(10)

# Understand the date type and shape 

df.shape

df.columns.tolist()

df.dtypes

#Preprocessing data(Missing_data)

df.isnull().sum()

#handle missing values

df.fillna({'income': df['income'].median()}, inplace=True)

df.fillna({'loan_amount': df['loan_amount'].median()}, inplace=True)

df.fillna({'credit_history': df['credit_history'].mode()[0]}, inplace=True)

df.isnull().sum()

#Summary  statistics

df.describe()

#Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns

# Set seaborn theme

sns.set(style='whitegrid')

# Create figure and axes

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# First plot

sns.histplot(df['income'], kde=True, bins=30, ax=axes[0])

axes[0].set_title("Income Distribution")

# Second plot

sns.histplot(df['loan_amount'], kde=True, bins=30, ax=axes[1])

axes[1].set_title("Loan Amount Distribution")

# Layout + Show

plt.tight_layout()

plt.show()

#Loan Tearm count plot

sns.countplot(x='term', data=df)

plt.title("Loan Term Distribution")

plt.xlabel("Loan Term(Month)")

plt.ylabel("Count")

plt.show()

#Credit History vs Defaulted

sns.countplot(x='credit_history', hue="defaulted", data=df)

plt.title("Credit_History vs Defaulted")

plt.xlabel("Credit_History")

plt.ylabel("Count")   

plt.legend(title="Defaulted")

plt.show()

#Correlation Heatmap

plt.figure(figsize=(8, 4))

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Feature Correlation Heatmap")

plt.show()	

#Convert Categorical Feature

# 36 - 0 and  60 - 1

df["term_binary"] = df["term"].apply(lambda x:1 if x==60 else 0)

# Create Derived Features (Optional but insightful)

import numpy as np

df["log_income"] = np.log(df["income"])

df["log_loan_amount"] = np.log(df["loan_amount"])

#Feature Selection (Modeling)

features = ["log_income","log_loan_amount", "credit_history"]

target = "defaulted"

#Featuring Scaling (Model Training)  - Using StandardScaler from sklearn

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler ()

scale_features = ["log_income", "log_loan_amount"]

df[scale_features] = scaler.fit_transform(df[scale_features])

df.head()

#Train Test split

from sklearn.model_selection import train_test_split

x = df[features]

y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#Build model pipeline

#import required classess from sklearn

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score

models = {

    "LogisticRegression":LogisticRegression(max_iter=1000, random_state=42),
    
    "DecisionTreeClassifier":DecisionTreeClassifier(random_state=42),
    
    "RandomForestClassifier":RandomForestClassifier(n_estimators=100, random_state=42)
    
}

for name, model in models.items():

    print(f"\nModel:{name}")
    
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier()

pipeline = Pipeline([

    ("classifier", model)
    
])

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

print(f"Accuracy_score: {accuracy_score(y_test, y_pred):.4f}")

print("Classification Report:")

print(classification_report(y_test, y_pred))

