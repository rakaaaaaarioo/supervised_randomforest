import streamlit as st
import pandas as pd
import pickle

# Load model Random Forest
model_file = 'rffruit.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Load dataset
file_path = 'fruit.xlsx'
df = pd.read_excel(file_path)
X = df[['diameter', 'weight', 'red', 'green', 'blue']]  # Fitur
y = df['name']  # Label target

# Mapping label ke kelas secara manual
label_to_class = {'grapefruit': 0, 'orange': 1}  # Pastikan sesuai
class_to_label = {v: k for k, v in label_to_class.items()}  # Membalik mapping

# Fungsi untuk prediksi
def predict_fruit(features):
    prediction_class = model.predict([features])[0]  # Prediksi kelas
    prediction_label = class_to_label[prediction_class]  # Mapping ke label
    return prediction_label, prediction_class

# Konfigurasi Streamlit
st.title("Aplikasi Prediksi Buah Menggunakan Random Forest")
st.write("Masukkan fitur buah untuk memprediksi jenis buah.")

# Input pengguna
input_features = []
for col in X.columns:
    value = st.number_input(f"Masukkan nilai untuk {col}:")
    input_features.append(value)

# Prediksi
if st.button("Prediksi"):
    label, class_index = predict_fruit(input_features)
    st.success(f"Model memprediksi jenis buah: {label} (Cluster: {class_index})")
