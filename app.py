import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- SETTING HALAMAN ---
st.set_page_config(page_title="Prediksi Kasus", layout="wide")
st.title("📊 Aplikasi Prediksi Kasus per Kecamatan")

# --- LOAD ASSET ---
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 🔁 Ganti dengan model asli kamu jika sudah ada:
# model = joblib.load("model.pkl")
model = RandomForestClassifier()  # dummy saja agar tidak error

# --- INPUT FILE ---
st.markdown("### 1. Upload Dataset Excel")
uploaded_file = st.file_uploader("Upload file `.xlsx` berisi data kecamatan", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("📄 Data yang berhasil dimuat:")
        st.dataframe(df)

        # --- PREPROCESSING ---
        X = df.select_dtypes(include=[np.number])
        X_scaled = scaler.transform(X)

        # --- PREDIKSI ---
        # Jika model asli belum tersedia, dummy prediksi acak:
        dummy_predictions = np.random.randint(0, len(label_encoder.classes_), size=len(X_scaled))
        y_pred = label_encoder.inverse_transform(dummy_predictions)

        df["Prediksi Kasus"] = y_pred

        # --- OUTPUT ---
        st.markdown("### 2. Hasil Prediksi")
        st.dataframe(df)

        # --- DOWNLOAD ---
        st.markdown("### 3. Unduh Hasil")
        output_xlsx = df.to_excel("hasil_prediksi.xlsx", index=False)
        with open("hasil_prediksi.xlsx", "rb") as f:
            st.download_button("📥 Download Excel", f, "hasil_prediksi.xlsx")

    except Exception as e:
        st.error(f"Gagal memproses file: {e}")

else:
    st.info("Silakan upload file Excel untuk mulai.")
