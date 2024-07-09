import streamlit as st
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Fungsi untuk memuat data
def load_data():
    with open('hungarian.data', encoding='Latin1') as file:
        lines = [line.strip() for line in file]
    data = itertools.takewhile(
        lambda x: len(x) == 76,
        (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
    )
    df = pd.DataFrame.from_records(data)
    df.replace('-9', np.nan, inplace=True)
    df.drop(columns=df.columns[-1], inplace=True)
    selected_columns = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]
    df = df.iloc[:, selected_columns]
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                  'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df = df.astype(float)
    df.fillna(df.mean(), inplace=True)
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

# Fungsi untuk melatih model
def train_models(df):
    X = df.drop(columns=['num'])
    y = df['num']
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    xgb_model = XGBClassifier(objective="binary:logistic", random_state=42)
    xgb_model.fit(X_train, y_train)
    
    return scaler, knn, xgb_model

# Fungsi untuk membuat prediksi
def make_prediction(model, scaler, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    return model.predict(input_data)

# Main function
def main():
    st.title("Heart Disease Prediction")
    
    # Memuat data dan melatih model
    df = load_data()
    scaler, knn, xgb_model = train_models(df)
    
    st.write("### Input Attributes")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    sex = st.selectbox("Sex", options=[0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3])
    
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    if st.button("Predict with KNN"):
        result_knn = make_prediction(knn, scaler, input_data)
        st.write(f"KNN Prediction: {'Heart Disease' if result_knn[0] > 0 else 'No Heart Disease'}")
        
    if st.button("Predict with XGBoost"):
        result_xgb = make_prediction(xgb_model, scaler, input_data)
        st.write(f"XGBoost Prediction: {'Heart Disease' if result_xgb[0] > 0 else 'No Heart Disease'}")

if __name__ == '__main__':
    main()
