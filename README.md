# ❤️ Heart Disease Prediction App

This is a **Machine Learning-powered Web App** built with **Python, Scikit-learn, and Streamlit** that predicts whether a person has **Heart Disease** based on medical attributes.

The app allows users to:

* Explore the **dataset** with a preview and statistics
* Check the **model performance** (accuracy on train/test sets)
* Enter **patient medical details** and get a prediction (Heart Disease or Not)
* Learn about the project via the **About page**

---

## 🚀 Features

* **Interactive Web App** using Streamlit
* **Logistic Regression** model trained on the Heart Disease dataset
* **Sidebar Navigation** with:

  * 🏠 **Home** → Dataset info & model performance
  * 🔍 **Predict** → Enter patient data & get prediction
  * ℹ️ **About** → App info and disclaimer
* Displays **training & testing accuracy**
* Clean UI with success/error messages for predictions

---

## 📂 Project Structure

```
├── heart_disease_app.py      # Streamlit app
├── heart_disease_data.csv    # Dataset
├── Heart_disease_prediction.ipynb # Jupyter Notebook (exploration & training)
└── README.md                 # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/Tom-1508/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Create virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *(Create a `requirements.txt` with:)*

   ```
   streamlit
   pandas
   numpy
   scikit-learn
   ```

4. **Run the app**

   ```bash
   streamlit run heart_disease_app.py
   ```

5. Open the app in your browser: **[http://localhost:8501](http://localhost:8501)**

---

## 📊 Dataset

* The dataset used is **heart\_disease\_data.csv**, containing patient medical attributes.
* Target variable:

  * `1` → Person **has Heart Disease**
  * `0` → Person **does not have Heart Disease**

---

## 📈 Model

* **Algorithm:** Logistic Regression
* **Training Accuracy:** \~ varies depending on dataset
* **Testing Accuracy:** \~ varies depending on dataset

---

## ⚠️ Disclaimer

This app is created for **educational purposes only** and should **not** be used for actual medical diagnosis.

---

## 👨‍💻 Author

**Tamal Majumdar**

---
