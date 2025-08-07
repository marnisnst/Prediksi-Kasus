
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, Flatten, Embedding, MultiHeadAttention, LayerNormalization, Add, Concatenate, Lambda
from tensorflow.keras.callbacks import EarlyStopping

# === FOLDER SETUP ===
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# === CONSTANTS ===
MODEL_PATH = "model/model.h5"
SCALER_PATH = "model/scaler.save"
ENCODER_PATH = "model/label_encoder.save"
DATA_PATH = "data/dataset.xlsx"

# === FUNCTION: TRAIN MODEL ===
def train_and_save_model(df):
    le = LabelEncoder()
    df['kecamatan_encoded'] = le.fit_transform(df['kecamatan'])

    month_map = {
        'Januari': '01', 'Februari': '02', 'Maret': '03', 'April': '04', 'Mei': '05', 'Juni': '06',
        'Juli': '07', 'Agustus': '08', 'September': '09', 'Oktober': '10', 'November': '11', 'Desember': '12'
    }
    df['month_num'] = df['bulan'].map(month_map)
    df['Date'] = pd.to_datetime(df['tahun'].astype(str) + '-' + df['month_num'])
    df = df.sort_values(by='Date')
    df['jumlah kasus'] = pd.to_numeric(df['jumlah kasus'], errors='coerce').fillna(0)

    scaler = MinMaxScaler()
    df['kasus_scaled'] = scaler.fit_transform(df[['jumlah kasus']])

    def create_windowed(ts_data, kec_data, y_data, input_steps=12, horizon=3):
        X_ts, X_kec, y_out = [], [], []
        for i in range(len(ts_data) - input_steps - horizon + 1):
            X_ts.append(ts_data[i:i+input_steps])
            X_kec.append(kec_data[i+input_steps])
            y_out.append(y_data[i+input_steps:i+input_steps+horizon].flatten())
        return np.array(X_ts), np.array(X_kec).reshape(-1,1), np.array(y_out)

    X_ts, X_kec, y = create_windowed(
        df[['kasus_scaled']].values,
        df['kecamatan_encoded'].values,
        df['kasus_scaled'].values,
        input_steps=12, horizon=3
    )

    num_kec = len(le.classes_)
    ts_input = Input(shape=(12,1), name="TimeSeriesInput")
    kec_input = Input(shape=(1,), name="KecamatanInput")

    kec_emb = Embedding(input_dim=num_kec, output_dim=8)(kec_input)
    kec_rep = Lambda(lambda x: tf.repeat(x, repeats=12, axis=1))(kec_emb)

    x = Concatenate(axis=-1)([ts_input, kec_rep])
    x = Lambda(lambda z: z + tf.sin(tf.cast(tf.range(12), tf.float32))[..., tf.newaxis])(x)

    gru = GRU(32, return_sequences=True)(x)
    lstm = LSTM(32, return_sequences=True)(gru)
    res = Add()([gru, lstm])
    att = MultiHeadAttention(num_heads=2, key_dim=2)(res, res)
    norm = LayerNormalization(epsilon=1e-6)(att)

    flat = Flatten()(norm)
    dense = Dense(64, activation='relu')(flat)
    output = Dense(3)(dense)

    model = Model(inputs=[ts_input, kec_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')

    model.fit([X_ts, X_kec], y, epochs=30, batch_size=16, verbose=0)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)
    df.to_excel(DATA_PATH, index=False)

# === LOAD COMPONENTS ===
def load_all():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    df = pd.read_excel(DATA_PATH)
    return model, scaler, le, df

# === WEB APP ===
st.set_page_config("Prediksi Kasus")
st.title("📊 Prediksi Kasus 1-3 Bulan")

if not os.path.exists(DATA_PATH):
    st.warning("🚨 Dataset belum tersedia. Upload untuk pertama kali.")

uploaded = st.file_uploader("Upload Data Bulanan (Excel)", type=['xlsx'])
if uploaded:
    df_new = pd.read_excel(uploaded)
    try:
        df_old = pd.read_excel(DATA_PATH)
        df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates()
    except:
        df_combined = df_new

    with st.spinner("🔁 Melatih ulang model..."):
        train_and_save_model(df_combined)
    st.success("✅ Data ditambahkan dan model diperbarui!")

# === PREDIKSI ===
if os.path.exists(MODEL_PATH):
    model, scaler, le, df = load_all()

    selected_kec = st.selectbox("Pilih Kecamatan", df['kecamatan'].unique())
    selected_month = st.selectbox("Tampilkan prediksi bulan ke-", [1,2,3])

    df_kec = df[df['kecamatan'] == selected_kec].copy()
    df_kec = df_kec.sort_values(by='Date')
    df_kec['jumlah kasus'] = pd.to_numeric(df_kec['jumlah kasus'], errors='coerce').fillna(0)
    df_kec['kasus_scaled'] = scaler.transform(df_kec[['jumlah kasus']])

    if len(df_kec) >= 12:
        x_ts = np.array(df_kec['kasus_scaled'].values[-12:]).reshape(1,12,1)
        x_kec = le.transform([selected_kec]).reshape(1,1)
        y_pred_scaled = model.predict([x_ts, x_kec])
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

        st.success(f"📅 Prediksi bulan ke-{selected_month} untuk {selected_kec}: {y_pred[selected_month-1]:.2f} kasus")

        fig, ax = plt.subplots()
        ax.plot(range(1,13), df_kec['jumlah kasus'].values[-12:], label='12 Bulan Terakhir')
        ax.plot(range(13,16), y_pred, label='Prediksi 3 Bulan')
        ax.axvline(12.5, linestyle='--', color='gray')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Data tidak cukup untuk prediksi (butuh minimal 12 bulan per kecamatan).")
print("Memulai aplikasi...")

import streamlit as st
import pandas as pd
import joblib

print("Import selesai")

# Coba load scaler dan encoder
try:
    scaler = joblib.load("scaler.pkl")
    print("Berhasil load scaler")
except Exception as e:
    print("Gagal load scaler:", e)

try:
    le = joblib.load("label_encoder.pkl")
    print("Berhasil load label encoder")
except Exception as e:
    print("Gagal load label encoder:", e)
