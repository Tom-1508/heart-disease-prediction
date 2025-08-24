import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------
# Load the dataset
# ------------------------------
@st.cache_data
def load_data():
    heart_data = pd.read_csv("heart_disease_data.csv")
    return heart_data

heart_data = load_data()

# Split features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict", "ℹ️ About"])

# ------------------------------
# Home Page
# ------------------------------
if page == "🏠 Home":
    st.title("❤️ Heart Disease Prediction App")
    st.write(
        """
        This is a **Machine Learning-powered Web App** that predicts 
        whether a person has **Heart Disease** based on medical attributes.  
        
        Navigate to the **Predict** page to try it out!
        """
    )
    with st.expander("📊 Dataset Preview"):
        st.write(heart_data.head())
        st.write("Shape:", heart_data.shape)
        st.write("Target distribution:", heart_data['target'].value_counts())

    # Model performance
    st.subheader("📈 Model Performance")
    train_acc = accuracy_score(model.predict(X_train), Y_train)
    test_acc = accuracy_score(model.predict(X_test), Y_test)
    st.metric("Training Accuracy", f"{train_acc:.2f}")
    st.metric("Test Accuracy", f"{test_acc:.2f}")

# ------------------------------
# Prediction Page
# ------------------------------
elif page == "🔍 Predict":
    st.title("🔍 Heart Disease Prediction")

    st.subheader("Enter Patient Data:")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
    restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.number_input("ST depression induced by exercise", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of peak exercise ST segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])

    # Prediction
    if st.button("Predict"):
        input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                      exang, oldpeak, slope, ca, thal)

        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        prediction = model.predict(input_data_as_numpy_array)

        if prediction[0] == 0:
            st.success("✅ The person does **NOT** have Heart Disease")
        else:
            st.error("⚠️ The person **HAS** Heart Disease")

# ------------------------------
# About Page
# ------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.write(
        """
        - Built with **Python, Scikit-learn, and Streamlit**  
        - Model: **Logistic Regression**  
        - Dataset: Heart Disease dataset (CSV)  
        - Created for demonstration of a simple ML project with deployment  
        
        **Author:** Tamal Majumdar
        """
    )
    st.info("⚠️ Disclaimer: This app is for educational purposes only and not for medical use.")