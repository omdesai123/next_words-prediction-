import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Next-Word Predictor",
    page_icon="🔮",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0b0c10;
    color: #e8e6e1;
}

/* ── Background grid texture ── */
[data-testid="stAppViewContainer"] {
    background-color: #0b0c10;
    background-image:
        linear-gradient(rgba(99,230,190,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,230,190,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }

/* ── Main card ── */
.main-card {
    background: rgba(16, 18, 25, 0.92);
    border: 1px solid rgba(99,230,190,0.18);
    border-radius: 2px;
    padding: 2.5rem 2.8rem 2rem;
    margin-top: 1rem;
    box-shadow: 0 0 60px rgba(99,230,190,0.06), 0 0 0 1px rgba(99,230,190,0.06);
}

/* ── Title ── */
.title-block {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 0.25rem;
}
.title-main {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.03em;
    color: #e8e6e1;
    line-height: 1;
}
.title-accent {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #63e6be;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border: 1px solid rgba(99,230,190,0.35);
    padding: 3px 8px;
    border-radius: 2px;
    vertical-align: middle;
}
.subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: rgba(232,230,225,0.38);
    letter-spacing: 0.08em;
    margin-bottom: 2.2rem;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid rgba(99,230,190,0.12);
    margin: 1.8rem 0;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #63e6be;
    margin-bottom: 0.5rem;
}

/* ── Textarea ── */
textarea {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.88rem !important;
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid rgba(99,230,190,0.22) !important;
    border-radius: 2px !important;
    color: #e8e6e1 !important;
    caret-color: #63e6be !important;
    transition: border-color 0.2s ease !important;
}
textarea:focus {
    border-color: rgba(99,230,190,0.55) !important;
    box-shadow: 0 0 0 3px rgba(99,230,190,0.07) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div { padding-top: 0 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #63e6be !important;
    border-color: #63e6be !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] {
    background: rgba(99,230,190,0.18) !important;
}

/* ── Button ── */
[data-testid="stButton"] button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid #63e6be !important;
    color: #63e6be !important;
    border-radius: 2px !important;
    padding: 0.55rem 2rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
[data-testid="stButton"] button:hover {
    background: rgba(99,230,190,0.1) !important;
    box-shadow: 0 0 20px rgba(99,230,190,0.15) !important;
}
[data-testid="stButton"] button:active {
    background: rgba(99,230,190,0.2) !important;
}

/* ── Prediction cards ── */
.pred-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 1rem;
}
.pred-card {
    display: flex;
    align-items: center;
    gap: 16px;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(99,230,190,0.12);
    border-radius: 2px;
    padding: 12px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.pred-card:hover { border-color: rgba(99,230,190,0.3); }
.pred-card .bar {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    background: rgba(99,230,190,0.06);
    transition: width 0.4s ease;
}
.pred-card .rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: rgba(99,230,190,0.5);
    letter-spacing: 0.1em;
    width: 22px;
    flex-shrink: 0;
    z-index: 1;
}
.pred-card .word {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 1.25rem;
    color: #e8e6e1;
    flex: 1;
    z-index: 1;
}
.pred-card.top-1 .word { color: #63e6be; }
.pred-card .pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: rgba(232,230,225,0.45);
    z-index: 1;
}
.pred-card.top-1 .pct { color: rgba(99,230,190,0.7); }

/* ── Error box ── */
.err-box {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #ff6b6b;
    border: 1px solid rgba(255,107,107,0.3);
    background: rgba(255,107,107,0.06);
    border-radius: 2px;
    padding: 12px 16px;
    margin-top: 1rem;
}

/* ── Status chip ── */
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #63e6be;
    opacity: 0.7;
}
.status-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #63e6be;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Composed sentence ── */
.composed {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: rgba(232,230,225,0.6);
    background: rgba(0,0,0,0.25);
    border-left: 2px solid rgba(99,230,190,0.4);
    padding: 10px 14px;
    border-radius: 0 2px 2px 0;
    margin-top: 1rem;
    word-break: break-word;
}
.composed .highlight { color: #63e6be; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    import h5py, json

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_lens.pkl", "rb") as f:
        max_len = pickle.load(f)

    # Strip quantization_config from all layer configs in the .h5 file
    def strip_quant(obj):
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for v in obj.values():
                strip_quant(v)
        elif isinstance(obj, list):
            for item in obj:
                strip_quant(item)

    with h5py.File("lstm_model.h5", "r+") as f:
        if "model_config" in f.attrs:
            cfg = json.loads(f.attrs["model_config"])
            strip_quant(cfg)
            f.attrs["model_config"] = json.dumps(cfg)

    model = tf.keras.models.load_model("lstm_model.h5", compile=False)
    return tokenizer, max_len, model


# ── Prediction logic ──────────────────────────────────────────────────────────
def predict_next_words(text: str, tokenizer, max_len: int, model, top_n: int = 5):
    seq = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(seq, maxlen=max_len, padding="pre")
    preds = model.predict(padded, verbose=0)[0]          # shape: (vocab_size,)
    top_indices = np.argsort(preds)[::-1][:top_n]
    results = []
    for idx in top_indices:
        word = tokenizer.index_word.get(idx, None)
        if word and word not in ("<UNK>", "[UNK]", "<unk>"):
            results.append((word, float(preds[idx])))
    return results


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="title-block">
  <span class="title-main">Next-Word Predictor</span>
  <span class="title-accent">LSTM</span>
</div>
<div class="subtitle">// language model · sequence prediction · v1.0</div>
""", unsafe_allow_html=True)

# Load artifacts
with st.spinner("Initialising model…"):
    try:
        tokenizer, max_len, model = load_artifacts()
        loaded_ok = True
    except Exception as e:
        loaded_ok = False
        load_err = str(e)

if loaded_ok:
    st.markdown(f'<div class="status-chip"><span class="status-dot"></span>Model ready · vocab {len(tokenizer.word_index):,} · seq {max_len}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="err-box">⚠ Failed to load model artifacts<br>{load_err}</div>', unsafe_allow_html=True)
    st.stop()

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Input
st.markdown('<div class="section-label">Seed text</div>', unsafe_allow_html=True)
seed_text = st.text_area(
    label="seed",
    label_visibility="collapsed",
    placeholder="Enter a sentence or phrase to continue…",
    height=110,
    key="seed_input",
)

col_slider, col_btn = st.columns([2, 1], gap="medium")
with col_slider:
    st.markdown('<div class="section-label">Top-N predictions</div>', unsafe_allow_html=True)
    top_n = st.slider("top_n", min_value=1, max_value=10, value=5, label_visibility="collapsed")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⟶  Predict", use_container_width=True)

# Prediction
if predict_btn:
    raw = seed_text.strip()
    if not raw:
        st.markdown('<div class="err-box">⚠ Please enter some seed text before predicting.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Running inference…"):
            try:
                results = predict_next_words(raw, tokenizer, max_len, model, top_n=top_n)
            except Exception as e:
                results = None
                infer_err = str(e)

        if results is None:
            st.markdown(f'<div class="err-box">⚠ Inference failed<br>{infer_err}</div>', unsafe_allow_html=True)
        elif not results:
            st.markdown('<div class="err-box">⚠ No valid predictions returned. Try a different input.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Predictions</div>', unsafe_allow_html=True)

            # Normalise probabilities among top-n for bar widths
            total = sum(p for _, p in results)
            cards_html = '<div class="pred-grid">'
            for i, (word, prob) in enumerate(results):
                pct = (prob / total * 100) if total > 0 else 0
                rank_label = f"#{i+1:02d}"
                top_cls = " top-1" if i == 0 else ""
                cards_html += f"""
                <div class="pred-card{top_cls}">
                    <div class="bar" style="width:{pct:.1f}%"></div>
                    <div class="rank">{rank_label}</div>
                    <div class="word">{word}</div>
                    <div class="pct">{prob*100:.2f}%</div>
                </div>"""
            cards_html += "</div>"
            st.markdown(cards_html, unsafe_allow_html=True)

            # Composed sentence preview
            best_word = results[0][0]
            composed = raw + " <span class='highlight'>" + best_word + "</span>"
            st.markdown(f'<div class="composed">{composed}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close main-card
