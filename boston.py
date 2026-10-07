import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Page configuration
st.set_page_config(page_title="Boston Housing Price Prediction", layout="wide")

# Title and Description
st.title("Boston Housing Regression Project")
st.write("This app predicts the regional median home values (MEDV) using local housing features.")

# Show image from the notebook
st.image("https://www.rickberk.com/images/xl/Autumn-Twilight-Boston.jpg", width=700)

# Load Data
@st.cache_data
def load_data():
    # Make sure 'boston.csv' is in the same directory
    df = pd.read_csv("boston.csv")
    return df

try:
    df = load_data()
    
    # Sidebar options
    st.sidebar.header("Options")
    show_data = st.sidebar.checkbox("Show Raw Data", False)
    
    if show_data:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
    # Exploratory Data Analysis section
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Summary**")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write(df.describe())
        
    with col2:
        st.write("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
    # Model Training
    st.subheader("Model Training & Prediction")
    
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]
    
    test_size = st.sidebar.slider("Test Size Ratio", 0.1, 0.4, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    st.write(f"**Model R² Score:** {r2:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    
    # Custom Prediction Section
    st.subheader("Predict MEDV for Custom Inputs")
    st.write("Adjust the features below to predict the median home value:")
    
    input_data = {}
    cols = st.columns(3)
    
    for i, col_name in enumerate(X.columns):
        min_val = float(df[col_name].min())
        max_val = float(df[col_name].max())
        mean_val = float(df[col_name].mean())
        
        with cols[i % 3]:
            input_data[col_name] = st.number_input(
                label=col_name, 
                min_value=min_val, 
                max_value=max_val, 
                value=mean_val
            )
            
    input_df = pd.DataFrame([input_data])
    
    if st.button("Predict House Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Median House Value (MEDV): **${prediction * 1000:,.2f}** ({prediction:.2f} in $1000s)")

except FileNotFoundError:
    st.error("The file 'boston.csv' was not found. Please place it in the same directory as this script.")