"""
==========================================================================
DASHBOARD PREDIKSI OBESITAS SISWA SMA/SMK
==========================================================================
Dashboard interaktif untuk prediksi risiko obesitas siswa
menggunakan Logistic Regression sebagai model utama
==========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Prediksi Obesitas Siswa",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS STYLING
# ==========================================
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e88e5 0%, #0d47a1 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2d5a87;
        margin-bottom: 1rem;
    }
    
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-obesitas {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
    }
    
    .result-tidak-obesitas {
        background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
        color: white;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
    
    /* Info box */
    .info-box {
        background: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Divider */
    .section-divider {
        border-top: 2px solid #e9ecef;
        margin: 2rem 0;
    }
    
    /* Risk level colors */
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #fd7e14; font-weight: bold; }
    .risk-critical { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL & ARTIFACTS
# ==========================================
@st.cache_resource
def load_model():
    """Load model dan artifacts dari file pickle"""
    try:
        # Coba beberapa lokasi file yang mungkin
        possible_paths = [
            "model_data.pkl",
            r"C:\Users\ANISETUS B. MANALU\kelompok_06\models\model_data.pkl",
            "./model_data.pkl",
            "models/model_data.pkl"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            return None, "File model_data.pkl tidak ditemukan. Pastikan file ada di folder yang sama dengan aplikasi."
            
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            
        return model_data, None
        
    except Exception as e:
        return None, f"Terjadi kesalahan saat memuat model: {str(e)}"

# ==========================================
# MAPPING UNTUK INPUT
# ==========================================
MAPPING_TIDUR = {
    "< 5 jam": 4.0,
    "5-6 jam": 5.5,
    "7-8 jam": 7.5,
    "> 8 jam": 9.0
}

MAPPING_MAKAN = {
    "1 kali": 1.0,
    "2 kali": 2.0,
    "3 kali": 3.0,
    "> 3 kali": 4.0
}

MAPPING_JAJAN = {
    "0-2 kali": 1.0,
    "3-5 kali": 4.0,
    "6-10 kali": 8.0,
    "> 10 kali": 12.0
}

MAPPING_FASTFOOD = {
    "0-2 kali": 1,
    "3-5 kali": 4,
    "> 5 kali": 7
}

MAPPING_MINUMAN = {
    "0-2 gelas": 1,
    "3-5 gelas": 4,
    "6-10 gelas": 8,
    "> 10 gelas": 12
}

MAPPING_MAKAN_MALAM = {
    "0 kali": 0.0,
    "1 kali": 1.0,
    "2-3 kali": 2.5,
    "4 kali": 4.0,
    "> 4 kali": 6.0
}

MAPPING_AKTIVITAS = {
    "Sangat Rendah": 1,
    "Rendah": 2,
    "Sedang": 3,
    "Tinggi": 4,
    "Sangat Tinggi": 5
}

MAPPING_STRES = {
    "Sangat Rendah": 1,
    "Rendah": 2,
    "Sedang": 3,
    "Tinggi": 4,
    "Sangat Tinggi": 5
}

MAPPING_TEMAN = {
    "Sangat Rendah": 1,
    "Rendah": 2,
    "Sedang": 3,
    "Tinggi": 4,
    "Sangat Tinggi": 5
}

MAPPING_MAKAN_STRES = {
    "Sangat Jarang": 1,
    "Jarang": 2,
    "Kadang-kadang": 3,
    "Sering": 4,
    "Sangat Sering": 5
}

MAPPING_VIDEO_MAKANAN = {
    "0-2 jam": 1.0,
    "3-5 jam": 4.0,
    "6-10 jam": 8.0,
    "> 10 jam": 12.0
}

# ==========================================
# FUNGSI PREDIKSI - LOGISTIC REGRESSION SAJA
# ==========================================
def predict_obesity_logreg(input_data, model_data):
    """
    Prediksi menggunakan Logistic Regression sebagai model utama
    (Random Forest hanya untuk perbandingan, bukan bagian dari prediksi final)
    """
    logreg = model_data['logreg']
    scaler = model_data['scaler']
    imputer = model_data['imputer']
    features = model_data['features']
    threshold_lr = model_data['threshold_lr']  # Gunakan threshold optimal dari training
    
    # Buat DataFrame
    data = pd.DataFrame([input_data], columns=features)
    
    # Imputasi dan scaling
    data_imputed = imputer.transform(data)
    data_scaled = scaler.transform(data_imputed)
    
    # Prediksi dengan Logistic Regression
    prob_lr = logreg.predict_proba(data_scaled)[0, 1]
    pred_lr = 1 if prob_lr >= threshold_lr else 0
    
    return {
        'probability': prob_lr,
        'prediction': pred_lr,
        'threshold': threshold_lr
    }

def get_random_forest_info(input_data, model_data):
    """Hanya untuk informasi perbandingan, bukan untuk prediksi final"""
    rf = model_data['rf']
    scaler = model_data['scaler']
    imputer = model_data['imputer']
    features = model_data['features']
    threshold_rf = model_data['threshold_rf']
    
    data = pd.DataFrame([input_data], columns=features)
    data_imputed = imputer.transform(data)
    data_scaled = scaler.transform(data_imputed)
    
    prob_rf = rf.predict_proba(data_scaled)[0, 1]
    pred_rf = 1 if prob_rf >= threshold_rf else 0
    
    return {
        'probability': prob_rf,
        'prediction': pred_rf,
        'threshold': threshold_rf
    }

def get_risk_level(probability, threshold=0.5396):
    """Menentukan level risiko berdasarkan probabilitas"""
    if probability < threshold - 0.2:
        return "RENDAH", "#28a745"
    elif probability < threshold:
        return "SEDANG", "#ffc107"
    elif probability < threshold + 0.2:
        return "TINGGI", "#fd7e14"
    else:
        return "SANGAT TINGGI", "#dc3545"

def create_gauge_chart(probability, title="Probabilitas Obesitas"):
    """Membuat gauge chart untuk visualisasi probabilitas"""
    prob_percent = probability * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': '%', 'font': {'size': 26}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1e88e5"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 50], 'color': '#fff3cd'},
                {'range': [50, 70], 'color': '#ffe5d0'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.8,
                'value': prob_percent
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_radar_chart(input_values, labels):
    """Membuat radar chart untuk visualisasi faktor risiko"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=input_values,
        theta=labels,
        fill='toself',
        fillcolor='rgba(30, 136, 229, 0.3)',
        line=dict(color='#1e88e5', width=2),
        name='Profil Siswa'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Sistem Prediksi Risiko Obesitas Siswa</h1>
        <p>Menggunakan Logistic Regression sebagai Model Utama</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_data, error = load_model()
    
    if error:
        st.error(f"⚠️ {error}")
        st.info("""
        **Penyelesaian masalah:**
        1. Pastikan file `model_data.pkl` ada di folder yang sama
        2. Atau jalankan script training terlebih dahulu
        3. Atau letakkan file di: `C:\\Users\\ANISETUS B. MANALU\\kelompok_06\\models\\model_data.pkl`
        """)
        return
    
    # ==========================================
    # SIDEBAR - INPUT FORM
    # ==========================================
    with st.sidebar:
        st.markdown("## 📝 Data Siswa")
        st.markdown("---")
        
        # Data Dasar
        st.markdown("### 👤 Data Dasar")
        usia = st.number_input("Usia (tahun)", min_value=10, max_value=25, value=16, step=1)
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        keluarga_obesitas = st.selectbox("Riwayat Keluarga Obesitas", ["Tidak", "Iya"])
        
        st.markdown("---")
        
        # Pola Makan
        st.markdown("### 🍽️ Pola Makan (per minggu)")
        makan_per_hari = st.selectbox("Frekuensi Makan/Hari", list(MAPPING_MAKAN.keys()), index=2)
        minuman_manis = st.selectbox("Minuman Manis", list(MAPPING_MINUMAN.keys()), index=1)
        fastfood = st.selectbox("Fast Food", list(MAPPING_FASTFOOD.keys()), index=0)
        jajan = st.selectbox("Jajan", list(MAPPING_JAJAN.keys()), index=1)
        makan_malam = st.selectbox("Makan Setelah Jam 21:00", list(MAPPING_MAKAN_MALAM.keys()), index=1)
        makan_stres = st.selectbox("Makan Karena Stres", list(MAPPING_MAKAN_STRES.keys()), index=1)
        video_makanan = st.selectbox("Menonton Video Makanan (jam)", list(MAPPING_VIDEO_MAKANAN.keys()), index=1)

        st.markdown("---")
        
        # Aktivitas & Kesehatan
        st.markdown("### 🏃 Aktivitas & Kesehatan")
        aktivitas_fisik = st.selectbox("Tingkat Aktivitas Fisik", list(MAPPING_AKTIVITAS.keys()), index=2)
        durasi_tidur = st.selectbox("Durasi Tidur/Hari", list(MAPPING_TIDUR.keys()), index=2)
        
        st.markdown("---")
        
        # Faktor Psikososial
        st.markdown("### 🧠 Faktor Psikososial")
        tingkat_stres = st.selectbox("Tingkat Stres", list(MAPPING_STRES.keys()), index=2)
        pengaruh_teman = st.selectbox("Pengaruh Teman", list(MAPPING_TEMAN.keys()), index=2)
        
        st.markdown("---")
        
        # Informasi Model
        with st.expander("ℹ️ **Informasi Model**"):
            st.info(f"""
            **Model Utama:** Logistic Regression
            **Threshold Optimal:** {model_data.get('threshold_lr', 0.5396):.4f}
            **Fitur:** {len(model_data['features'])} variabel
            **Alasan Pemilihan:**
            - Performa lebih konsisten pada data tidak seimbang
            - Interpretasi koefisien yang jelas
            - Probabilitas yang stabil
            """)
        
        # Tombol Prediksi
        predict_button = st.button("🔍 Prediksi Risiko Obesitas", type="primary", use_container_width=True)
    
    # ==========================================
    # MAIN CONTENT
    # ==========================================
    if predict_button:
        # Encode input values
        jk_encode = 1 if jenis_kelamin == "Laki-laki" else 0
        keluarga_encode = 1 if keluarga_obesitas == "Iya" else 0
        
        # Build input data
        input_data = [
            usia,
            jk_encode,
            MAPPING_MAKAN[makan_per_hari],
            MAPPING_MINUMAN[minuman_manis],
            MAPPING_FASTFOOD[fastfood],
            MAPPING_JAJAN[jajan],
            MAPPING_AKTIVITAS[aktivitas_fisik],
            MAPPING_TIDUR[durasi_tidur],
            MAPPING_STRES[tingkat_stres],
            MAPPING_TEMAN[pengaruh_teman],
            keluarga_encode,
            MAPPING_MAKAN_MALAM[makan_malam],
            MAPPING_AKTIVITAS[aktivitas_fisik],  # aktivitas_harian sama dengan aktivitas_fisik
            MAPPING_MAKAN_STRES[makan_stres],
            MAPPING_VIDEO_MAKANAN[video_makanan]
        ]
        
        # Predict dengan Logistic Regression (model utama)
        result_logreg = predict_obesity_logreg(input_data, model_data)
        
        # Untuk perbandingan saja (tidak digunakan dalam prediksi final)
        result_rf = get_random_forest_info(input_data, model_data)
        
        # Store in session state
        st.session_state['result_logreg'] = result_logreg
        st.session_state['result_rf'] = result_rf
        st.session_state['input_labels'] = {
            'usia': usia,
            'jenis_kelamin': jenis_kelamin,
            'keluarga_obesitas': keluarga_obesitas,
            'makan_per_hari': makan_per_hari,
            'minuman_manis': minuman_manis,
            'fastfood': fastfood,
            'jajan': jajan,
            'makan_malam': makan_malam,
            'makan_stres': makan_stres,
            'video_makanan': video_makanan,
            'aktivitas_fisik': aktivitas_fisik,
            'durasi_tidur': durasi_tidur,
            'tingkat_stres': tingkat_stres,
            'pengaruh_teman': pengaruh_teman
        }
    
    # Display results if available
    if 'result_logreg' in st.session_state:
        result_logreg = st.session_state['result_logreg']
        input_labels = st.session_state['input_labels']
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Hasil Prediksi", "📈 Analisis Detail", "💡 Rekomendasi"])
        
        # ==========================================
        # TAB 1: HASIL PREDIKSI
        # ==========================================
        with tab1:
            # Ambil hasil Logistic Regression (model utama)
            pred = result_logreg['prediction']
            prob = result_logreg['probability']
            threshold = result_logreg['threshold']
            
            # Tentukan tingkat risiko
            risk_level, risk_color = get_risk_level(prob, threshold)
            
            # Display result
            if pred == 1:
                st.markdown(f"""
                <div class="result-card result-obesitas">
                    <h2>⚠️ BERISIKO OBESITAS</h2>
                    <h3>Probabilitas: {prob*100:.1f}%</h3>
                    <p><b>Level Risiko:</b> <span style="color:{risk_color};">{risk_level}</span></p>
                    <p><b>Threshold Model:</b> {threshold:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-tidak-obesitas">
                    <h2>✅ TIDAK BERISIKO OBESITAS</h2>
                    <h3>Probabilitas: {prob*100:.1f}%</h3>
                    <p><b>Level Risiko:</b> <span style="color:{risk_color};">{risk_level}</span></p>
                    <p><b>Threshold Model:</b> {threshold:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Gauge Chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_gauge_chart(prob, "Logistic Regression (Model Utama)"),
                    use_container_width=True
                )
                st.markdown(f"""
                <div style="text-align: center">
                    <p><b>Prediksi:</b> {"Obesitas" if pred == 1 else "Tidak Obesitas"}</p>
                    <p><b>Threshold Optimal:</b> {threshold:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Untuk perbandingan saja
                result_rf = st.session_state['result_rf']
                st.plotly_chart(
                    create_gauge_chart(result_rf['probability'], "Random Forest (Pembanding)"),
                    use_container_width=True
                )
                st.markdown(f"""
                <div style="text-align: center">
                    <p><b>Prediksi:</b> {"Obesitas" if result_rf['prediction'] == 1 else "Tidak Obesitas"}</p>
                    <p><b>Threshold:</b> {result_rf['threshold']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Informasi Model
            st.markdown("---")
            st.markdown("### ℹ️ Alasan Pemilihan Model")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="info-box">
                    <h4>✅ Logistic Regression (Model Utama)</h4>
                    <p><b>Alasan pemilihan:</b></p>
                    <ul>
                        <li>Performanya lebih konsisten pada dataset tidak seimbang</li>
                        <li>Koefisien dapat diinterpretasikan dengan jelas</li>
                        <li>Hasil probabilitas yang stabil dan konsisten</li>
                        <li>AUC lebih tinggi dari Random Forest (0.94 vs 0.93)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-box">
                    <h4>⚠️ Mengapa Tidak Menggunakan Ensemble?</h4>
                    <p><b>Alasan ilmiah:</b></p>
                    <ul>
                        <li>Ensemble tidak meningkatkan performa signifikan</li>
                        <li>Threshold menjadi ambigu (LR: 0.5396, RF: 0.3967)</li>
                        <li>Probabilitas tidak konsisten antar model</li>
                        <li>Lebih mudah dipertahankan dalam laporan akademik</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # ==========================================
        # TAB 2: ANALISIS DETAIL
        # ==========================================
        with tab2:
            st.markdown("### 📊 Profil Gaya Hidup")
            
            # Data untuk visualisasi
            radar_labels = ['Pola Makan', 'Aktivitas Fisik', 'Kualitas Tidur', 'Stres', 'Lingkungan']
            radar_values = [
                (MAPPING_MINUMAN[input_labels['minuman_manis']] + 
                 MAPPING_FASTFOOD[input_labels['fastfood']] + 
                 MAPPING_JAJAN[input_labels['jajan']]) / 3,
                MAPPING_AKTIVITAS[input_labels['aktivitas_fisik']],
                5 if MAPPING_TIDUR[input_labels['durasi_tidur']] >= 7 else 3,
                MAPPING_STRES[input_labels['tingkat_stres']],
                MAPPING_TEMAN[input_labels['pengaruh_teman']]
            ]
            
            fig_radar = create_radar_chart(radar_values, radar_labels)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Ringkasan Data
            st.markdown("---")
            st.markdown("### 📋 Ringkasan Data Input")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**👤 Data Dasar**")
                st.write(f"• Usia: {input_labels['usia']} tahun")
                st.write(f"• Jenis Kelamin: {input_labels['jenis_kelamin']}")
                st.write(f"• Riwayat Keluarga: {input_labels['keluarga_obesitas']}")
            
            with col2:
                st.markdown("**🍽️ Pola Makan**")
                st.write(f"• Makan/Hari: {input_labels['makan_per_hari']}")
                st.write(f"• Minuman Manis: {input_labels['minuman_manis']}")
                st.write(f"• Fast Food: {input_labels['fastfood']}")
                st.write(f"• Jajan: {input_labels['jajan']}")
                st.write(f"• Video Makanan: {input_labels['video_makanan']}")
            
            with col3:
                st.markdown("**🏃 Aktivitas & Kesehatan**")
                st.write(f"• Aktivitas Fisik: {input_labels['aktivitas_fisik']}")
                st.write(f"• Durasi Tidur: {input_labels['durasi_tidur']}")
                st.write(f"• Tingkat Stres: {input_labels['tingkat_stres']}")
                st.write(f"• Pengaruh Teman: {input_labels['pengaruh_teman']}")
        
        # ==========================================
        # TAB 3: REKOMENDASI
        # ==========================================
        with tab3:
            st.markdown("### 💡 Rekomendasi Personal")
            
            recommendations = []
            
            # Pola Makan
            if MAPPING_MINUMAN[input_labels['minuman_manis']] > 4:
                recommendations.append((
                    "🥤", 
                    "Kurangi Minuman Manis", 
                    f"Anda mengonsumsi {input_labels['minuman_manis']} per minggu. Batasi maksimal 3-5 gelas per minggu."
                ))
            
            if MAPPING_FASTFOOD[input_labels['fastfood']] > 3:
                recommendations.append((
                    "🍔", 
                    "Kurangi Fast Food", 
                    f"Anda mengonsumsi {input_labels['fastfood']} per minggu. Batasi maksimal 1-2 kali per minggu."
                ))
            
            if MAPPING_JAJAN[input_labels['jajan']] > 4:
                recommendations.append((
                    "🍿", 
                    "Kontrol Jajan", 
                    f"Anda jajan {input_labels['jajan']} per minggu. Kurangi dan pilih camilan sehat."
                ))
            
            # Aktivitas
            if MAPPING_AKTIVITAS[input_labels['aktivitas_fisik']] < 3:
                recommendations.append((
                    "🏃", 
                    "Tingkatkan Aktivitas Fisik", 
                    "Aktivitas fisik Anda rendah. Targetkan olahraga minimal 3 kali per minggu."
                ))
            
            # Tidur
            if MAPPING_TIDUR[input_labels['durasi_tidur']] < 7:
                recommendations.append((
                    "😴", 
                    "Perbaiki Pola Tidur", 
                    f"Anda tidur {input_labels['durasi_tidur']}. Targetkan 7-8 jam per malam."
                ))
            
            # Stres
            if MAPPING_STRES[input_labels['tingkat_stres']] > 3:
                recommendations.append((
                    "🧘", 
                    "Kelola Stres", 
                    f"Tingkat stres Anda {input_labels['tingkat_stres']}. Coba meditasi atau teknik relaksasi."
                ))
            
            # Tampilkan rekomendasi
            if recommendations:
                for icon, title, desc in recommendations:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>{icon} {title}</h4>
                        <p>{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Gaya Hidup Baik!</h4>
                    <p>Berdasarkan data yang dimasukkan, gaya hidup Anda sudah cukup sehat. Pertahankan!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tips Umum
            st.markdown("---")
            st.markdown("### 📚 Tips Pencegahan Obesitas")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🥗 Nutrisi Sehat**
                - Konsumsi sayur dan buah setiap hari
                - Batasi makanan tinggi gula dan lemak
                - Minum air putih minimal 8 gelas/hari
                - Makan teratur 3x sehari
                
                **🏃 Aktivitas Fisik**
                - Olahraga minimal 30 menit, 3-5x/minggu
                - Kurangi waktu duduk/diam
                - Gunakan tangga daripada lift
                """)
            
            with col2:
                st.markdown("""
                **😴 Pola Hidup Sehat**
                - Tidur 7-8 jam setiap malam
                - Kelola stres dengan baik
                - Hindari makan larut malam
                - Batasi screen time berlebihan
                
                **🧠 Kesehatan Mental**
                - Jaga hubungan sosial positif
                - Cari hobi untuk mengisi waktu luang
                - Konsultasi jika stres berlebihan
                """)
    
    else:
        # Default view
        st.markdown("""
        <div class="info-box">
            <h4>👋 Selamat Datang di Sistem Prediksi Obesitas Siswa</h4>
            <p><b>Petunjuk penggunaan:</b></p>
            <ol>
                <li>Isi data siswa di sidebar kiri</li>
                <li>Klik tombol <b>"Prediksi Risiko Obesitas"</b></li>
                <li>Lihat hasil prediksi dan rekomendasi</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Informasi Model
        if model_data:
            st.markdown("### ⚙️ Spesifikasi Model")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Utama", "Logistic Regression")
            
            with col2:
                st.metric("Threshold Optimal", f"{model_data.get('threshold_lr', 0.5396):.4f}")
            
            with col3:
                st.metric("Jumlah Fitur", len(model_data['features']))

# ==========================================
# RUN APPLICATION
# ==========================================
if __name__ == "__main__":
    main()