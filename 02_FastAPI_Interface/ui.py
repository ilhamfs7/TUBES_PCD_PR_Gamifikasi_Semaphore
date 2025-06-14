import streamlit as st
import requests
from PIL import Image
import io

# URL Backend FastAPI
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide", page_title="Semaphore Game")

st.title("🎮 Semaphore Gesture Game")
st.write("Gunakan gerakan tubuhmu untuk mengetik kata yang ditampilkan di layar!")

# Layout Utama
col1, col2 = st.columns([3, 1]) # Kolom video lebih besar

with col2:
    st.header("Kontrol Game")
    
    difficulty = st.selectbox(
        "Pilih Tingkat Kesulitan:",
        ("mudah", "sedang", "sulit")
    )

    if st.button("Ubah Kesulitan"):
        try:
            requests.post(f"{BACKEND_URL}/set_difficulty/{difficulty}")
            st.success(f"Kesulitan diubah menjadi {difficulty}!")
        except requests.exceptions.ConnectionError:
            st.error("Tidak dapat terhubung ke server. Pastikan backend sudah berjalan.")

    if st.button("▶️ Mulai Latihan (Practice)"):
        try:
            requests.post(f"{BACKEND_URL}/start_game/practice")
            st.info("Mode latihan dimulai!")
        except requests.exceptions.ConnectionError:
            st.error("Tidak dapat terhubung ke server.")

    if st.button("🏆 Mulai Tantangan (Challenge)"):
        try:
            requests.post(f"{BACKEND_URL}/start_game/challenge")
            st.info("Mode tantangan dimulai!")
        except requests.exceptions.ConnectionError:
            st.error("Tidak dapat terhubung ke server.")
    
    st.header("Statistik Pemain")
    # Tempat untuk menampilkan statistik
    stats_placeholder = st.empty()

with col1:
    st.header("Video Feed")
    # Tempat untuk menampilkan video
    video_placeholder = st.empty()
    video_placeholder.info("Menunggu koneksi ke server video feed...")

# Coba ambil status game secara berkala (meskipun Streamlit akan re-run)
try:
    state_response = requests.get(f"{BACKEND_URL}/game_state")
    if state_response.status_code == 200:
        state = state_response.json()
        stats_placeholder.json(state['player_stats'])
    else:
        stats_placeholder.warning("Gagal mengambil status game.")
except requests.exceptions.ConnectionError:
    stats_placeholder.error("Tidak dapat terhubung ke server.")

# Menampilkan video stream
video_placeholder.image(f"{BACKEND_URL}/video_feed", use_column_width=True)
st.caption("Jika video tidak muncul, pastikan backend berjalan dan webcam Anda diizinkan.")