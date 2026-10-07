# 🏠 Boston Housing Price Prediction

A regression and data analysis project that estimates regional median home values using local housing features from the Boston Housing dataset.

The project combines a **Jupyter Notebook for Exploratory Data Analysis (EDA), outlier handling, and regression model comparison** with an **interactive Streamlit application** where users can explore the data, adjust the train/test split, and predict median home values from custom feature inputs.

## 📌 Project Overview

The main objective of this project is to estimate **MEDV (Median Home Value)** using a collection of local housing and regional features.

The project explores questions such as:

- Which housing features are most strongly related to median home value?
- How do room count and LSTAT relate to MEDV?
- How does Linear Regression perform on the dataset?
- Which regression algorithm provides the best predictive performance?
- How do model results change after handling extreme values?
- Can the trained model be turned into an interactive prediction application?

## 📊 Dataset

The project uses:

```text
boston.csv
```

The notebook contains **506 rows and 14 columns**.

The available columns are:

```text
CRIM
ZN
INDUS
CHAS
NOX
RM
AGE
DIS
RAD
TAX
PTRATIO
B
LSTAT
MEDV
```

`MEDV` is used as the prediction target, while the remaining 13 columns are used as input features.

The notebook reports **no missing values** in the dataset.

## 📈 Target Variable

The prediction target is:

```text
MEDV
```

The notebook visualizes MEDV as **Median Value in $1,000s**.

For example:

```text
MEDV = 25
```

represents an estimated median home value of approximately:

```text
$25,000
```

The Streamlit application also displays predictions both in `$1,000s` and as a formatted dollar value.

## 🔎 Exploratory Data Analysis

The Jupyter Notebook (`BostonRegression.ipynb`) includes:

- Dataset preview
- Shape analysis
- Data-type inspection
- Column inspection
- Descriptive statistics
- Missing-value analysis
- Correlation matrix
- Correlation heatmap
- MEDV distribution
- RM vs. MEDV scatter plot
- LSTAT vs. MEDV scatter plot
- Feature correlation ranking against MEDV
- Outlier filtering
- Regression modeling
- Multi-model performance comparison
- Residual analysis with Yellowbrick

## 🔗 Correlation Analysis

The notebook calculates correlations between the features and `MEDV`.

Two of the strongest relationships highlighted by the analysis are:

- `RM` and `MEDV`: approximately **0.695**
- `LSTAT` and `MEDV`: approximately **-0.738**

The notebook specifically visualizes both relationships using scatter plots.

This shows that some housing variables have substantially stronger relationships with median value than others.

## 🧹 Outlier Handling

Before the full model comparison, the notebook calculates the **97th percentile** of the data and removes extreme observations from several features:

```text
CRIM
ZN
RM
DIS
PTRATIO
B
LSTAT
```

This preprocessing step is used in the notebook modeling workflow to reduce the effect of extreme feature values.

## 🤖 Linear Regression Baseline

The notebook first trains a standard `LinearRegression` model.

The data is split using:

```text
80% Training
20% Testing
random_state = 42
```

The baseline Linear Regression model produces approximately:

| Metric | Result |
|---|---:|
| R² | **0.750** |
| RMSE | **3.827** |

Because MEDV is expressed in thousands of dollars, the RMSE corresponds to roughly **3.83 units in $1,000s**.

## 🧠 Regression Models Compared

The notebook then compares eight regression algorithms:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Extra Tree Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor
- XGBoost Regressor

Each model is evaluated using:

- R²
- RMSE
- MAE

## 🏆 Model Performance

The notebook reports the following model comparison:

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| **Gradient Boosting Regressor** | **0.922** | **2.141** | **1.735** |
| XGBoost Regressor | 0.874 | 2.714 | 2.030 |
| Ridge | 0.763 | 3.730 | 2.731 |
| Linear Regression | 0.750 | 3.827 | 2.887 |
| ElasticNet | 0.746 | 3.858 | 2.875 |
| Lasso | 0.734 | 3.949 | 2.960 |
| Extra Tree Regressor | 0.583 | 4.944 | 3.318 |
| K-Nearest Neighbors | 0.436 | 5.750 | 4.069 |

The notebook concludes that **Gradient Boosting Regressor** provides the best overall performance among the tested models.

### Best Model

```text
GradientBoostingRegressor
R²   ≈ 0.922
RMSE ≈ 2.14
MAE  ≈ 1.74
```

The model explains approximately **92% of the variation in MEDV** within the notebook's evaluation setup.

## 📉 Residual Analysis

The notebook also uses `Yellowbrick`'s `ResidualsPlot` to visualize the errors produced by the Linear Regression model.

Residual analysis helps evaluate whether prediction errors show systematic patterns rather than being randomly distributed.

## 🖥️ Streamlit Housing Predictor

The project also includes an interactive Streamlit application (`boston.py`).

Unlike the notebook's multi-model comparison, the Streamlit application focuses specifically on a **Linear Regression** model.

## 📊 Streamlit Features

The application includes:

- Optional raw-data preview
- Dataset row and column summary
- Descriptive statistics
- Correlation heatmap
- Adjustable test-size ratio
- Linear Regression training
- R² score
- RMSE
- Custom feature inputs
- Real-time MEDV prediction

## ⚙️ Adjustable Test Size

The sidebar allows users to change the test-set ratio from:

```text
0.10 to 0.40
```

with a default value of:

```text
0.20
```

The Linear Regression model is retrained using the selected train/test split.

## 🏡 Custom House Value Prediction

The application automatically creates an input field for every predictor column.

Each field uses the observed:

- Minimum value
- Maximum value
- Mean value

from the dataset.

After entering custom values and clicking:

```text
Predict House Price
```

the model generates an estimated MEDV.

The application displays the result in both forms:

```text
MEDV value in $1,000s
```

and:

```text
Estimated dollar value
```

## 📊 Application Workflow

```text
Boston Housing Data
        ↓
Exploratory Data Analysis
        ↓
Select Test Size
        ↓
Train Linear Regression
        ↓
Evaluate R² and RMSE
        ↓
Enter Custom Features
        ↓
Predict MEDV
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Scikit-learn
- XGBoost
- Yellowbrick
- Linear Regression
- Gradient Boosting
- Jupyter Notebook

## 🎯 Project Purpose

This project demonstrates practical skills in:

- Exploratory Data Analysis
- Data visualization
- Correlation analysis
- Outlier handling
- Regression modeling
- Machine learning model comparison
- Linear Regression
- Regularization methods
- Gradient Boosting
- XGBoost
- Model evaluation
- Residual analysis
- Interactive prediction applications
- Streamlit development
- 
Built with Python, Scikit-learn, and Streamlit to explore regression modeling and predict regional median home values. 🏠📊🤖
