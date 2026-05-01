
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import subprocess
import tempfile
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Musical Instrument Classifier",
    page_icon="🎵",
    layout="wide"
)

# ─── Constants ───────────────────────────────────────────────────────────────
SR        = 22050
DURATION  = 4
MEL_SHAPE = (128, 128)

CLASSES     = ['flu', 'gac', 'pia', 'sax', 'vio']
CLASS_NAMES = {
    'flu': 'Flute',
    'gac': 'Acoustic Guitar',
    'pia': 'Piano',
    'sax': 'Saxophone',
    'vio': 'Violin',
}
CLASS_EMOJI = {
    'flu': '🎵', 'gac': '🎸', 'pia': '🎹',
    'sax': '🎷', 'vio': '🎻',
}
CLASS_DESC = {
    'flu':  'Bright airy tone — woodwind family',
    'gac':  'Warm resonant pluck — string family',
    'pia':  'Rich harmonic range — keyboard family',
    'sax':  'Reedy expressive tone — woodwind family',
    'vio':  'Bowed string with vibrato — string family',
}

AUDIO_PATH = '/content/MusicInstruments/musical_instruments/audio_wav'

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.metric-card {
    background: #111118; border: 1px solid #2a2a3a;
    border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center;
}
.metric-label {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #888; margin-bottom: 0.3rem;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #7B61FF; line-height: 1; }
.metric-sub   { font-size: 0.78rem; color: #666; margin-top: 0.3rem; }

.class-card {
    background: #111118; border: 1px solid #2a2a3a;
    border-radius: 10px; padding: 1rem 1.2rem;
    display: flex; align-items: center; gap: 0.8rem;
}
.class-emoji { font-size: 1.6rem; }
.class-name  { font-weight: 700; font-size: 1rem; color: #e0e0f0; }
.class-desc  { font-size: 0.78rem; color: #777; }

.result-banner {
    background: linear-gradient(135deg, #1a1428 0%, #111118 100%);
    border: 1px solid #7B61FF55; border-radius: 14px;
    padding: 1.6rem 2rem; margin: 1.2rem 0;
}
.result-instrument { font-size: 2.4rem; font-weight: 800; color: #fff; line-height: 1.1; }
.result-desc       { font-size: 0.9rem; color: #999; margin-top: 0.3rem; }
.result-conf       { font-family: 'Space Mono', monospace; font-size: 1.4rem; color: #7B61FF; font-weight: 700; margin-top: 0.6rem; }

.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #7B61FF; margin: 1.8rem 0 0.7rem 0;
}

.how-box {
    background: #0d0d14; border-left: 3px solid #7B61FF;
    border-radius: 0 8px 8px 0; padding: 1rem 1.2rem;
    font-size: 0.88rem; color: #aaa; line-height: 1.7;
}

.conf-bar-wrap  { background: #1a1a2a; border-radius: 6px; height: 10px; overflow: hidden; margin-top: 4px; }
.conf-bar-fill  { height: 10px; border-radius: 6px; background: linear-gradient(90deg, #7B61FF, #b09fff); }

.stem-card {
    background: #111118; border: 1px solid #2a2a3a;
    border-radius: 12px; padding: 1.2rem;
}

.info-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.info-table th {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: #666;
    padding: 0.5rem 0.8rem; border-bottom: 1px solid #2a2a3a; text-align: left;
}
.info-table td { padding: 0.6rem 0.8rem; border-bottom: 1px solid #1a1a28; color: #ccc; }
.info-table tr:last-child td { border-bottom: none; }

h1 { font-family: 'Fantasy', sans-serif !important; font-weight: 1000 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
for key, default in [("page", "predict"), ("model", None), ("le", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('cnn_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, le
    except Exception:
        return None, None

# ─── Audio → mel input ────────────────────────────────────────────────────────
def audio_to_mel_input(file_path):
    y, sr = librosa.load(file_path, sr=SR, duration=DURATION, mono=True)
    target = SR * DURATION
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = Image.fromarray(mel_db).resize(MEL_SHAPE, Image.BILINEAR)
    arr    = np.array(img, dtype=np.float32)
    arr    = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr[np.newaxis, ..., np.newaxis], mel_db, y, sr

# ─── Demucs 6-stem separation — returns only guitar & piano ──────────────────
def separate_stems(file_path, out_dir="/tmp/demucs_out"):
    os.makedirs(out_dir, exist_ok=True)
    result = subprocess.run(
        ['python', '-m', 'demucs', '-n', 'htdemucs_6s', '-o', out_dir, file_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        st.error(f"Demucs error: {result.stderr[-500:]}")
        return {}

    song_name = os.path.splitext(os.path.basename(file_path))[0]

    # htdemucs_6s saves under htdemucs_6s/<songname>/
    stem_dir = os.path.join(out_dir, 'htdemucs_6s', song_name)
    if not os.path.isdir(stem_dir):
        st.error(f"Expected stem folder not found: {stem_dir}")
        return {}

    # Only keep guitar and piano — the two stems our CNN understands
    target_stems = {'guitar', 'piano'}
    stems = {}
    for f in os.listdir(stem_dir):
        if f.endswith('.wav'):
            stem_name = f.replace('.wav', '')
            if stem_name in target_stems:
                stems[stem_name] = os.path.join(stem_dir, f)

    return stems

# ─── Dataset info ─────────────────────────────────────────────────────────────
@st.cache_data
def get_dataset_info():
    info = {}
    total = 0
    if os.path.exists(AUDIO_PATH):
        for cls in CLASSES:
            path = os.path.join(AUDIO_PATH, cls)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith('.wav')]
                info[cls] = len(files)
                total += len(files)
            else:
                info[cls] = 0
        return info, total, True
    demo = {c: 300 for c in CLASSES}
    return demo, 1500, False

@st.cache_data
def get_sample_spectrograms():
    specs = {}
    if not os.path.exists(AUDIO_PATH):
        return specs
    for cls in CLASSES:
        path = os.path.join(AUDIO_PATH, cls)
        if not os.path.exists(path):
            continue
        wavs = [f for f in os.listdir(path) if f.endswith('.wav')]
        if not wavs:
            continue
        try:
            y, sr  = librosa.load(os.path.join(path, wavs[0]), duration=DURATION, sr=SR, mono=True)
            mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            specs[cls] = (librosa.power_to_db(mel, ref=np.max), sr)
        except Exception:
            pass
    return specs

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎵")
    st.markdown("### Musical Instrument Classifier")
    st.divider()

    st.markdown('<p class="section-header">Navigate</p>', unsafe_allow_html=True)
    for page_id, label in [("predict", "🎯  Predict"), ("detect", "🎛️  Detect Instruments"), ("info", "📊  Dataset Info"), ("classification", "📋  Classification")]:
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.page == page_id else "secondary"):
            st.session_state.page = page_id
            st.rerun()

    st.divider()
    st.markdown('<p class="section-header">Instruments</p>', unsafe_allow_html=True)
    for cls in CLASSES:
        st.markdown(
            f'<div class="class-card"><span class="class-emoji">{CLASS_EMOJI[cls]}</span>'
            f'<div><div class="class-name">{CLASS_NAMES[cls]}</div>'
            f'<div class="class-desc">{CLASS_DESC[cls]}</div></div></div><br>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-header">How it works</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="how-box">'
        '1. Your audio → Mel-Spectrogram image<br>'
        '2. CNN reads the image<br>'
        '3. Softmax picks the instrument<br>'
        '4. Confidence scores returned'
        '</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Built with librosa · TensorFlow · Streamlit · Demucs")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "predict":

    st.markdown("# 🎯 Predict Instrument")
    st.markdown("Upload any instrument audio clip → CNN predicts which instrument it is")
    st.divider()

    model, le = load_model()
    if model is None:
        st.warning("⚠️ Model not found (`cnn_model.h5`). Train the CNN first (Cell 9), then re-run the app.")
    else:
        st.success("✅ Model loaded and ready")

    st.markdown('<p class="section-header">Upload a sound clip</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a .wav, .mp3, or .ogg file",
        type=["wav", "mp3", "ogg"],
        label_visibility="collapsed",
        key="predict_upload"
    )

    if uploaded:
        st.audio(uploaded)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(uploaded.read())
        tmp.flush()

        with st.spinner("Analysing audio…"):
            mel_input, mel_db, y_audio, sr = audio_to_mel_input(tmp.name)

        if model is not None:
            with st.spinner("Running CNN…"):
                probs     = model.predict(mel_input, verbose=0)[0]
                pred_idx  = int(np.argmax(probs))
                pred_cls  = le.classes_[pred_idx]
                pred_name = CLASS_NAMES.get(pred_cls, pred_cls)
                pred_conf = float(probs[pred_idx])

            st.markdown(
                f'<div class="result-banner">'
                f'<div class="result-instrument">{CLASS_EMOJI.get(pred_cls,"🎵")} {pred_name}</div>'
                f'<div class="result-desc">{CLASS_DESC.get(pred_cls,"")}</div>'
                f'<div class="result-conf">{pred_conf:.1%} confidence</div>'
                f'</div>', unsafe_allow_html=True)

            st.markdown('<p class="section-header">Confidence per instrument</p>', unsafe_allow_html=True)
            for i, cls in enumerate(le.classes_):
                p = float(probs[i])
                bar_color = "#7B61FF" if cls == pred_cls else "#2a2a3a"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">'
                    f'<span style="width:130px;font-size:0.85rem;color:#ccc;">{CLASS_EMOJI.get(cls,"")} {CLASS_NAMES.get(cls,cls)}</span>'
                    f'<div class="conf-bar-wrap" style="flex:1;"><div class="conf-bar-fill" style="width:{p*100:.1f}%;background:{bar_color};"></div></div>'
                    f'<span style="width:46px;text-align:right;font-family:Space Mono,monospace;font-size:0.78rem;color:#999;">{p:.1%}</span>'
                    f'</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header">Mel-Spectrogram of your clip</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3))
        fig.patch.set_facecolor('#0d0d14')
        ax.set_facecolor('#0d0d14')
        img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel-Spectrogram (your audio)', fontsize=11, color='#ccc')
        ax.tick_params(colors='#777')
        for spine in ax.spines.values(): spine.set_edgecolor('#2a2a3a')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<p class="section-header">Waveform</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(9, 2))
        fig2.patch.set_facecolor('#0d0d14')
        ax2.set_facecolor('#0d0d14')
        librosa.display.waveshow(y_audio, sr=sr, alpha=0.8, color='#7B61FF', ax=ax2)
        ax2.set_title('Waveform', fontsize=11, color='#ccc')
        ax2.tick_params(colors='#777')
        for spine in ax2.spines.values(): spine.set_edgecolor('#2a2a3a')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DETECT INSTRUMENTS  (Demucs 6-stem → guitar & piano only)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "detect":

    st.markdown("# 🎛️ Detect Instruments in a Song")
    st.markdown(
        "Upload a full song → **Demucs htdemucs_6s** separates it into 6 stems "
        "→ only **🎸 Guitar** and **🎹 Piano** stems are kept and classified by the CNN"
    )
    st.divider()

    # Check demucs installed
    demucs_ok = subprocess.run(['python', '-m', 'demucs', '--help'],
                               capture_output=True).returncode == 0
    if not demucs_ok:
        st.error("Demucs is not installed. Run this in a Colab cell first:")
        st.code("!pip install demucs", language="bash")
        st.stop()

    model, le = load_model()
    if model is None:
        st.warning("⚠️ Model not found. Train the CNN first (Cell 9).")

    st.markdown('<p class="section-header">Upload a song</p>', unsafe_allow_html=True)
    song_file = st.file_uploader(
        "Choose a .wav or .mp3 file (any full song works)",
        type=["wav", "mp3"],
        label_visibility="collapsed",
        key="detect_upload"
    )

    if song_file:
        st.audio(song_file)

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(song_file.read())
        tmp.flush()

        # ── Separate with htdemucs_6s, keep only guitar & piano ──
        with st.spinner("Separating stems with Demucs htdemucs_6s… (~60–120 seconds)"):
            stems = separate_stems(tmp.name)

        if not stems:
            st.error("No guitar or piano stems found. The song may not contain these instruments, or Demucs failed.")
            st.stop()

        st.success(f"Extracted {len(stems)} stem(s): {', '.join(stems.keys())}")
        st.divider()

        # ── Classify guitar and piano stems ──
        st.markdown('<p class="section-header">Stem analysis — Guitar & Piano only</p>', unsafe_allow_html=True)

        stem_display = {
            'guitar': ('🎸', 'Guitar stem'),
            'piano':  ('🎹', 'Piano stem'),
        }

        cols = st.columns(len(stems))
        for col, (stem_name, stem_path) in zip(cols, stems.items()):
            with col:
                emoji, label = stem_display.get(stem_name, ('🎵', stem_name.capitalize()))
                st.markdown(f"### {emoji} {label}")
                st.audio(stem_path)

                if model is not None:
                    with st.spinner("CNN…"):
                        try:
                            mel_input, mel_db, _, sr = audio_to_mel_input(stem_path)
                            probs     = model.predict(mel_input, verbose=0)[0]
                            pred_idx  = int(np.argmax(probs))
                            pred_cls  = le.classes_[pred_idx]
                            pred_conf = float(probs[pred_idx])

                            st.markdown(
                                f'<div class="result-banner" style="padding:1rem;">'
                                f'<div style="font-size:1.4rem;font-weight:800;color:#fff;">'
                                f'{CLASS_EMOJI.get(pred_cls,"🎵")} {CLASS_NAMES.get(pred_cls, pred_cls)}</div>'
                                f'<div style="font-family:Space Mono,monospace;color:#7B61FF;margin-top:0.4rem;">'
                                f'{pred_conf:.1%}</div></div>',
                                unsafe_allow_html=True)

                            # Confidence bars — all 5 classes
                            for i, cls in enumerate(le.classes_):
                                p = float(probs[i])
                                bar_color = "#7B61FF" if cls == pred_cls else "#2a2a3a"
                                st.markdown(
                                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">'
                                    f'<span style="width:22px;font-size:0.8rem;">{CLASS_EMOJI.get(cls,"")}</span>'
                                    f'<div class="conf-bar-wrap" style="flex:1;height:6px;">'
                                    f'<div class="conf-bar-fill" style="width:{p*100:.1f}%;height:6px;background:{bar_color};"></div></div>'
                                    f'<span style="width:38px;font-size:0.7rem;color:#777;font-family:Space Mono,monospace;">{p:.0%}</span>'
                                    f'</div>', unsafe_allow_html=True)

                            # Spectrogram thumbnail
                            fig, ax = plt.subplots(figsize=(4, 2))
                            fig.patch.set_facecolor('#0d0d14')
                            ax.set_facecolor('#0d0d14')
                            librosa.display.specshow(mel_db, sr=sr, ax=ax, cmap='magma')
                            ax.axis('off')
                            plt.tight_layout(pad=0)
                            st.pyplot(fig)
                            plt.close()

                        except Exception as e:
                            st.error(f"Error: {e}")

        st.divider()
        st.caption("Stems separated by Demucs (htdemucs_6s model) · Only guitar & piano stems classified by CNN")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET INFO
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "info":

    st.markdown("# 📊 Dataset Info")
    st.markdown("Overview of the Musical Instruments dataset used to train the CNN")
    st.divider()

    file_counts, total_files, dataset_found = get_dataset_info()

    if not dataset_found:
        st.info("Dataset not found — showing demo values.", icon="ℹ️")

    col1, col2, col3, col4 = st.columns(4)
    for col, val, label, sub in [
        (col1, str(total_files), "Total audio files", "WAV format"),
        (col2, str(len(CLASSES)),  "Classes",           "instrument types"),
        (col3, str(total_files // len(CLASSES)), "Files / class", "avg, balanced"),
        (col4, f"{DURATION}s",     "Clip duration",     "fixed length"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{val}</div><div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True)

    st.markdown('<p class="section-header">Files per class</p>', unsafe_allow_html=True)
    max_count = max(file_counts.values()) if file_counts else 1
    rows = ""
    for cls in CLASSES:
        n = file_counts.get(cls, 0)
        pct = n / total_files * 100 if total_files else 0
        bw  = n / max_count * 100
        rows += (
            f'<tr><td>{CLASS_EMOJI[cls]} {CLASS_NAMES[cls]}</td>'
            f'<td style="font-family:Space Mono,monospace;color:#7B61FF;">{n}</td>'
            f'<td>{pct:.1f}%</td>'
            f'<td style="width:200px;"><div class="conf-bar-wrap">'
            f'<div class="conf-bar-fill" style="width:{bw:.0f}%;"></div></div></td></tr>'
        )
    st.markdown(
        f'<table class="info-table"><tr><th>Instrument</th><th>Files</th><th>Share</th><th>Distribution</th></tr>{rows}</table>',
        unsafe_allow_html=True)

    st.markdown('<p class="section-header">Model & feature spec</p>', unsafe_allow_html=True)
    specs = [
        ("Input",               "128 x 128 x 1 Mel-Spectrogram"),
        ("Sample rate",         f"{SR:,} Hz"),
        ("Clip length",         f"{DURATION} seconds"),
        ("Mel bins",            "128"),
        ("CNN architecture",    "3x Conv blocks (32-64-128) + Dense head"),
        ("Regularisation",      "BatchNorm + Dropout 0.25/0.5"),
        ("Optimizer",           "Adam lr 3e-4, ReduceLROnPlateau"),
        ("Loss",                "Sparse categorical cross-entropy"),
        ("Train / Val / Test",  "64% / 16% / 20%"),
        ("Output",              f"Softmax over {len(CLASSES)} classes"),
    ]
    spec_rows = "".join(
        f'<tr><td style="color:#888;width:220px;">{k}</td>'
        f'<td style="font-family:Space Mono,monospace;font-size:0.82rem;">{v}</td></tr>'
        for k, v in specs)
    st.markdown(
        f'<table class="info-table"><tr><th>Parameter</th><th>Value</th></tr>{spec_rows}</table>',
        unsafe_allow_html=True)

    st.markdown('<p class="section-header">Sample mel-spectrograms</p>', unsafe_allow_html=True)
    specs_data = get_sample_spectrograms()

    if specs_data:
        fig, axes = plt.subplots(1, len(CLASSES), figsize=(18, 4))
        fig.patch.set_facecolor('#0d0d14')
        for i, cls in enumerate(CLASSES):
            ax = axes[i]
            ax.set_facecolor('#0d0d14')
            if cls in specs_data:
                mel_db, sr = specs_data[cls]
                librosa.display.specshow(mel_db, ax=ax, cmap='magma', sr=sr)
            ax.set_title(f'{CLASS_EMOJI[cls]} {CLASS_NAMES[cls]}', fontsize=10, color='#ccc', pad=6)
            ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run Cells 4–6 in Colab to populate sample spectrograms.", icon="📁")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "classification":

    st.markdown("# 📈 Classification Results")
    st.markdown("CNN evaluation on the held-out test set (20% of dataset)")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    top_metrics = [
        (col1, "86.7%", "Test Accuracy",  "CNN on held-out set"),
        (col2, "0.4066", "Test Loss",     "cross-entropy"),
        (col3, "0.87",   "Macro F1",      "avg across 5 classes"),
        (col4, "700",    "Test samples",  "140 per class"),
    ]
    for col, val, label, sub in top_metrics:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{val}</div><div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True)

    st.markdown("")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_column_width=True)
        else:
            st.info("confusion_matrix.png not found. Run Cell 11 in your Colab notebook to generate it.", icon="📁")

    with img_col2:
        st.markdown('<p class="section-header">Training History</p>', unsafe_allow_html=True)
        if os.path.exists("training_history.png"):
            st.image("training_history.png", use_column_width=True)
        else:
            st.info("training_history.png not found. Run Cell 10 in your Colab notebook to generate it.", icon="📁")

    st.markdown('<p class="section-header">Classification Report</p>', unsafe_allow_html=True)
    report_data = [
        ("🎵 Flute",          "0.90", "0.81", "0.85", "140"),
        ("🎸 Acoustic Guitar", "0.85", "0.94", "0.89", "140"),
        ("🎹 Piano",           "0.85", "0.94", "0.89", "140"),
        ("🎷 Saxophone",       "0.89", "0.84", "0.86", "140"),
        ("🎻 Violin",          "0.86", "0.81", "0.83", "140"),
    ]

    header = '<table class="info-table" style="width:100%;"><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>'
    rows = ""
    for cls_name, prec, rec, f1, sup in report_data:
        f1_val = float(f1)
        bar_w = int(f1_val * 100)
        bar_html = f'<div class="conf-bar-wrap" style="width:80px;display:inline-block;vertical-align:middle;margin-left:6px;"><div class="conf-bar-fill" style="width:{bar_w}%;"></div></div>'
        rows += f'<tr><td style="font-weight:600;color:#e0e0f0;">{cls_name}</td><td style="font-family:Space Mono,monospace;">{prec}</td><td style="font-family:Space Mono,monospace;">{rec}</td><td style="font-family:Space Mono,monospace;">{f1} {bar_html}</td><td style="font-family:Space Mono,monospace;color:#888;">{sup}</td></tr>'

    avg_row = '<tr style="border-top:1px solid #7B61FF44;"><td style="color:#7B61FF;font-weight:700;">Weighted avg</td><td style="font-family:Space Mono,monospace;color:#7B61FF;">0.87</td><td style="font-family:Space Mono,monospace;color:#7B61FF;">0.87</td><td style="font-family:Space Mono,monospace;color:#7B61FF;">0.87</td><td style="font-family:Space Mono,monospace;color:#888;">700</td></tr>'
    st.markdown(header + rows + avg_row + "</table>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-header">Key observations</p>', unsafe_allow_html=True)
    obs_col1, obs_col2 = st.columns(2)
    with obs_col1:
        st.markdown('<div class="how-box">📌 <b>Best recall:</b> Acoustic Guitar & Piano (94%) — the model rarely misses these.<br><br>📌 <b>Hardest class:</b> Violin (81% recall) — most confused with other strings.<br><br>📌 <b>Training converged</b> around epoch 10; validation accuracy plateaued ~85%.</div>', unsafe_allow_html=True)
    with obs_col2:
        st.markdown('<div class="how-box">📌 <b>Overfitting visible</b> — train accuracy reaches ~99% while val stays ~85%. Consider more dropout or data augmentation.<br><br>📌 <b>Saxophone confusion:</b> 11 samples predicted as Violin — similar bowing/reed timbres.<br><br>📌 <b>Overall:</b> 86.7% on a 5-class balanced set is strong for a lightweight CNN.</div>', unsafe_allow_html=True)
