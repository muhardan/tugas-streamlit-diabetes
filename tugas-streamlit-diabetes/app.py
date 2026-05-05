import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")

# --- HEADER ---
st.title('Aplikasi Prediksi Penyakit Diabetes')
st.write('Dibuat oleh: Rama (Tugas Praktik Streamlit)')
st.markdown('---')

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('diabetes_model.sav', 'rb'))
        return model
    except FileNotFoundError:
        st.error("Model 'diabetes_model.sav' tidak ditemukan. Pastikan sudah diunggah ke GitHub.")
        return None

model = load_model()

# --- INPUT DATA ---
st.sidebar.header('Input Data Pasien')

def user_input_features():
    pregnancies = st.sidebar.number_input('Jumlah Kehamilan', min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input('Kadar Glukosa', min_value=0, max_value=200, value=100)
    blood_pressure = st.sidebar.number_input('Tekanan Darah', min_value=0, max_value=150, value=70)
    skin_thickness = st.sidebar.number_input('Ketebalan Kulit', min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input('Kadar Insulin', min_value=0, max_value=900, value=79)
    bmi = st.sidebar.number_input('Indeks Massa Tubuh (BMI)', min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input('Usia', min_value=1, max_value=120, value=30)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Menampilkan inputan
st.subheader('Data Pasien yang Diinput:')
st.write(input_df)

# --- PREDIKSI ---
if st.button('Cek Hasil Prediksi'):
    if model is not None:
        prediction = model.predict(input_df)
        
        st.markdown('---')
        st.subheader('Hasil Analisis:')
        if prediction[0] == 1:
            st.error('⚠️ Berdasarkan data tersebut, pasien terdeteksi terkena **DIABETES**.')
        else:
            st.success('✅ Berdasarkan data tersebut, pasien terdeteksi **TIDAK TERKENA DIABETES**.')
    else:
        st.warning("Silakan unggah model terlebih dahulu untuk melakukan prediksi.")

# --- FOOTER ---
st.markdown('---')
st.caption('Catatan: Aplikasi ini hanya untuk tujuan edukasi tugas kuliah.')