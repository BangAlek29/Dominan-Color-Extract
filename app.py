import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import hashlib

# --- Konfigurasi halaman ---
st.set_page_config(
    page_title="Dominant Color Extractor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .color-box {
        width: 100%;
        height: 80px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .color-info {
        text-align: center;
        font-size: 14px;
        margin-top: 5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Fungsi Utility ---
def get_image_hash(image):
    """Generate hash untuk mendeteksi perubahan gambar"""
    return hashlib.md5(np.array(image).tobytes()).hexdigest()

def extract_colors_kmeans(image, n_colors=5, resize_factor=0.25):
    """Extract dominant colors menggunakan KMeans"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    resized = cv2.resize(img_array, (int(w * resize_factor), int(h * resize_factor)))
    reshaped = resized.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(reshaped)

    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = Counter(labels)
    total = len(labels)

    color_info = []
    for i, color in enumerate(centers):
        pct = (counts[i] / total) * 100
        hex_code = '#{:02x}{:02x}{:02x}'.format(*color)
        color_info.append({
            'color': color,
            'hex': hex_code,
            'percentage': pct,
            'rgb': f"RGB({color[0]}, {color[1]}, {color[2]})"
        })

    return sorted(color_info, key=lambda x: x['percentage'], reverse=True)

def get_color_name(rgb):
    """Menentukan nama warna berdasarkan RGB"""
    r, g, b = rgb
    if r > 200 and g > 200 and b > 200:
        return "Putih/Terang"
    elif r < 50 and g < 50 and b < 50:
        return "Hitam/Gelap"
    elif r > g and r > b:
        return "Merah/Oranye" if g > 100 else "Merah"
    elif g > r and g > b:
        return "Hijau"
    elif b > r and b > g:
        return "Biru/Ungu" if r > 100 else "Biru"
    elif r > 150 and g > 150:
        return "Kuning"
    else:
        return "Abu-abu"

# --- Header ---
st.title("🎨 Dominant Color Extractor")

# --- Settings dalam expander ---
with st.expander("⚙️ Pengaturan"):
    col1, col2 = st.columns(2)
    with col1:
        n_colors = st.slider("Jumlah Warna", 3, 10, 5)
    with col2:
        resize_factor = st.slider("Kualitas Analisis", 0.1, 1.0, 0.3, step=0.1)

st.markdown("---")

# --- Upload Gambar ---
uploaded_file = st.file_uploader(
    "Pilih gambar", 
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    help="Format yang didukung: PNG, JPG, JPEG, BMP, TIFF"
)

if uploaded_file is not None:
    # Load dan convert gambar
    image = Image.open(uploaded_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Generate hash untuk cache busting
    image_hash = get_image_hash(image)
    
    # Simpan hash di session state untuk deteksi perubahan
    if 'last_image_hash' not in st.session_state or st.session_state.last_image_hash != image_hash:
        st.session_state.last_image_hash = image_hash
        # Clear cache ketika gambar berubah
        if 'cached_colors' in st.session_state:
            del st.session_state.cached_colors
    
    # Layout utama
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Gambar Original")
        st.image(image, use_container_width=True, caption=f"Resolusi: {image.size[0]} × {image.size[1]} pixels")
    
    with col2:
        st.subheader("Analisis Warna")
        
        # Cache hasil analisis berdasarkan parameter
        cache_key = f"{image_hash}_{n_colors}_{resize_factor}"
        
        if cache_key not in st.session_state:
            with st.spinner("Menganalisis warna dominan..."):
                try:
                    colors = extract_colors_kmeans(image, n_colors, resize_factor)
                    st.session_state[cache_key] = colors
                except Exception as e:
                    st.error(f"❌ Error dalam analisis: {str(e)}")
                    st.stop()
        else:
            colors = st.session_state[cache_key]
                
        # Grid layout untuk warna
        n_cols = min(3, len(colors))
        for i in range(0, len(colors), n_cols):
            cols = st.columns(n_cols)
            for j, col in enumerate(cols):
                if i + j < len(colors):
                    c = colors[i + j]
                    with col:
                        st.markdown(f"""
                            <div class="color-box" style="background-color: {c['hex']};"></div>
                            <div class="color-info">
                                <strong>{c['hex']}</strong><br>
                                <small>{c['percentage']:.1f}%</small><br>
                                <small>{get_color_name(c['color'])}</small>
                            </div>
                        """, unsafe_allow_html=True)
    
    # Tabel detail warna
    st.markdown("---")
    st.subheader("Detail Warna")
    
    # Buat dataframe
    df_data = []
    for i, c in enumerate(colors):
        df_data.append({
            'Rank': i + 1,
            'Hex Code': c['hex'],
            'RGB': c['rgb'],
            'Persentase': f"{c['percentage']:.1f}%",
            'Nama Warna': get_color_name(c['color'])
        })
    
    df = pd.DataFrame(df_data)
    
    # Tampilkan tabel dengan styling
    st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Hex Code": st.column_config.TextColumn("Hex Code", width="medium"),
            "RGB": st.column_config.TextColumn("RGB", width="medium"),
            "Persentase": st.column_config.TextColumn("Persentase", width="small"),
            "Nama Warna": st.column_config.TextColumn("Nama Warna", width="medium")
        }
    )
    
    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Salin Hex Code")
        hex_codes = ", ".join([c['hex'] for c in colors])
        st.code(hex_codes, language=None)

    with col2:
        st.markdown("##### Download Data Warna")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"color_analysis_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # Placeholder ketika tidak ada gambar
    st.info("👆 Upload gambar untuk memulai analisis warna dominan")
    
    # Contoh gambar atau instruksi
    st.markdown("""
    ### 📝 Cara Penggunaan:
    1. **Upload gambar** menggunakan tombol di atas
    2. **Atur pengaturan** jika diperlukan (opsional)
    3. **Lihat hasil** analisis warna dominan
    4. **Download hasil** dalam format CSV jika diperlukan
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "Made with ❤️<br>"
    "Created by <strong>Dzikri Bassyril</strong>"
    "</div>", 
    unsafe_allow_html=True
)