# Bank Loan Prediction App

This project involves building a Decision Tree model for predicting the approval of bank loans based on various customer attributes. The dataset contains information such as age, gender, income, loan amount, credit history, and more, which helps to predict whether a loan application will be approved or not.

## Table of Contents
- [Installation](#installation)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Modeling Process](#modeling-process)
  - [Step 1: Import Libraries and Dataset](#step-1-import-libraries-and-dataset)
  - [Step 2: Data Preprocessing](#step-2-data-preprocessing)
  - [Step 3: Data Partitioning](#step-3-data-partitioning)
  - [Step 4: Model Building](#step-4-model-building)
  - [Step 5: Model Evaluation](#step-5-model-evaluation)
  - [Step 6: Model Export](#step-6-model-export)
- [Model Performance](#model-performance)
- [License](#license)

## Installation

To run this project, you'll need to install the following libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn plotly
```

## Dataset Description

The dataset `Bank_Loan.csv` includes the following columns:
- **Loan_ID**: Unique identifier for each loan.
- **Age**: Age of the applicant.
- **Gender**: Gender of the applicant.
- **Married**: Marital status of the applicant.
- **Dependents**: Number of dependents.
- **Education**: Education level (Graduate/Not Graduate).
- **Self_Employed**: Whether the applicant is self-employed.
- **ApplicantIncome**: Monthly income of the applicant.
- **LoanAmount**: Loan amount requested.
- **Previous_Loan_Taken**: Whether the applicant has taken a previous loan.
- **Cibil_Score**: Credit score of the applicant.
- **Property_Area**: Type of property area (Urban/Rural).
- **Customer_Bandwith**: Bandwidth of the customer.
- **Tenure**: Duration of loan in months.
- **Loan_Status**: Target variable indicating whether the loan is approved (Yes/No).

## Project Structure

1. **Data Preprocessing**:
   - Removal of irrelevant variables.
   - Handling missing values.
   - Label encoding categorical features.
   - Feature selection for modeling.

2. **Modeling**:
   - Splitting data into training and testing sets.
   - Training a Decision Tree classifier.
   - Evaluating the model using accuracy, precision, recall, and F1-score.

3. **Visualization**:
   - Plotting the decision tree for better interpretability.
   - Using pie charts for data distribution visualization.

## Modeling Process

### Step 1: Import Libraries and Dataset

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
employee = pd.read_csv(r"C:\path_to_your_file\Bank_Loan.csv")
```

### Step 2: Data Preprocessing

- **Removal of Irrelevant Variables**:
  - Dropped `Loan_ID` as it does not influence the prediction.
- **Label Encoding**:
  - Converted categorical variables into numerical values for model compatibility.
- **EDA (Exploratory Data Analysis)**:
  - Visualized data distribution and checked for missing values.

### Step 3: Data Partitioning

Split the data into training and testing sets using a 70-30 split:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=231)
```

### Step 4: Model Building

Trained a Decision Tree classifier:

```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
```

### Step 5: Model Evaluation

Evaluated the model on both training and testing data using metrics like precision, recall, and F1-score.

### Step 6: Model Export

Exported the trained model using `pickle` for future predictions:

```python
import pickle
pickle.dump(dt2, open('model.pkl', 'wb'))
```

## Model Performance

- The decision tree model achieved **92% accuracy** on the training dataset and **93% accuracy** on the testing dataset after applying pruning techniques.
- Key performance metrics:
  - **Precision**: Measures the accuracy of positive predictions.
  - **Recall**: Measures the ability to capture all positive instances.
  - **F1-Score**: Balances precision and recall.
- The model can be further improved by experimenting with different hyperparameters and algorithms.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed.
```
