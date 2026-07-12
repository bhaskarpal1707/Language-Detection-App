import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="LinguaAI — Language Detection",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_models():
    model = pickle.load(open("language_detector.pkl", "rb"))
    vectorizer = pickle.load(open("count_vectorizer.pkl", "rb"))
    return model, vectorizer

model, cv = load_models()

# ============================================================
# Advanced Custom CSS
# ============================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(99,102,241,0.10), transparent 40%),
        radial-gradient(circle at 85% 0%, rgba(14,165,233,0.10), transparent 40%),
        linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1300px;
}

/* ---------------- Hero ---------------- */
.hero-wrap {
    text-align: center;
    padding: 30px 10px 10px 10px;
}

.hero-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 18px;
}

.main-title {
    font-family: 'Sora', sans-serif;
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #38bdf8, #818cf8);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 6s linear infinite;
    margin-bottom: 6px;
}

@keyframes shine {
    to { background-position: 200% center; }
}

.sub-title {
    font-size: 17px;
    color: #94a3b8;
    font-weight: 500;
    margin-bottom: 10px;
}

/* ---------------- Glass cards ---------------- */
.glass-card {
    border-radius: 20px;
    padding: 26px 28px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 22px;
    transition: all 0.25s ease;
}

.glass-card:hover {
    border-color: rgba(129,140,248,0.4);
    box-shadow: 0 12px 40px rgba(99,102,241,0.15);
}

.section-title {
    font-family: 'Sora', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}

.section-caption {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 18px;
}

/* ---------------- Result card ---------------- */
.result-card {
    border-radius: 24px;
    padding: 40px 30px;
    background: linear-gradient(135deg, #4f46e5, #0ea5e9);
    color: white;
    text-align: center;
    box-shadow: 0 20px 50px rgba(79,70,229,0.35);
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: "";
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.6; }
    50% { transform: scale(1.15); opacity: 1; }
}

.result-label {
    font-size: 14px;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.85;
    font-weight: 600;
    position: relative;
    z-index: 1;
}

.result-lang {
    font-family: 'Sora', sans-serif;
    font-size: 54px;
    font-weight: 800;
    margin: 8px 0 0 0;
    position: relative;
    z-index: 1;
}

/* ---------------- Metric pills ---------------- */
.metric-pill {
    border-radius: 16px;
    padding: 18px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

.metric-pill .val {
    font-family: 'Sora', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #e2e8f0;
}

.metric-pill .lbl {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ---------------- Top prediction rows ---------------- */
.rank-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 16px;
    border-radius: 14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 10px;
}

.rank-medal {
    font-size: 22px;
    width: 34px;
}

.rank-lang {
    font-weight: 700;
    color: #e2e8f0;
    width: 140px;
    font-size: 15px;
}

.rank-score {
    color: #a5b4fc;
    font-weight: 700;
    font-size: 14px;
    width: 60px;
    text-align: right;
}

/* ---------------- Language chips ---------------- */
.chip {
    display: inline-block;
    padding: 8px 16px;
    margin: 5px;
    border-radius: 999px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    color: #c7d2fe;
    font-size: 13px;
    font-weight: 600;
}

/* ---------------- Buttons ---------------- */
.stButton>button {
    border-radius: 12px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.2s ease;
}

.stButton>button:hover {
    border-color: #818cf8;
    color: #818cf8;
    transform: translateY(-1px);
}

/* ---------------- Footer ---------------- */
.footer-box {
    text-align: center;
    padding: 30px 20px;
    color: #64748b;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 20px;
}

.footer-box b { color: #cbd5e1; }

hr {
    border-color: rgba(255,255,255,0.08) !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:

    st.markdown("### 🌐 LinguaAI")
    st.caption("Multilingual Detection Engine")

    st.markdown("---")

    st.markdown("**👨‍💻 Developer**")
    st.write("Bhaskar Pal")
    st.markdown("[🌐 Portfolio](https://bhaskarpal1707.github.io/portfolio/)")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/bhaskar-pal-2k02/)")

    st.markdown("---")

    st.markdown("**⚙️ Model Details**")
    st.success("Multinomial Naive Bayes")
    st.info("CountVectorizer")
    st.metric("Supported Languages", "22")

    st.markdown("---")

    with st.expander("📚 Supported Languages", expanded=False):
        languages = [
            "Arabic", "Chinese", "Dutch", "English", "Estonian", "French",
            "Hindi", "Indonesian", "Japanese", "Korean", "Latin", "Persian",
            "Portugese", "Pushto", "Romanian", "Russian", "Spanish",
            "Swedish", "Tamil", "Thai", "Turkish", "Urdu"
        ]
        for lang in languages:
            st.write("•", lang)

# ============================================================
# Hero Section
# ============================================================
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-badge'>⚡ POWERED BY MACHINE LEARNING</div>
    <div class='main-title'>LinguaAI Language Detection</div>
    <div class='sub-title'>Instantly identify text across <b>22 languages</b> with real-time confidence scoring</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ============================================================
# Example Sentences
# ============================================================
examples = {
    "English": "Hello! Welcome to the AI Language Detection System. I hope you have a wonderful day.",
    "French": "Bonjour tout le monde. Comment allez-vous aujourd'hui ?",
    "Spanish": "Hola amigos. Bienvenidos a este sistema de detección de idiomas.",
    "Hindi": "नमस्ते, आप कैसे हैं? मुझे मशीन लर्निंग बहुत पसंद है।",
    "Japanese": "こんにちは。今日はとてもいい天気ですね。",
    "Chinese": "你好，欢迎使用人工智能语言检测系统。",
    "Arabic": "مرحبا بكم في نظام اكتشاف اللغة باستخدام الذكاء الاصطناعي.",
    "Russian": "Привет! Добро пожаловать в систему определения языка."
}

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ============================================================
# Input Card
# ============================================================
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>✍️ Enter Text</div>", unsafe_allow_html=True)
st.markdown("<div class='section-caption'>Type or paste any sentence, or try a quick example below.</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("🇬🇧 English", use_container_width=True):
        st.session_state.input_text = examples["English"]
    if st.button("🇫🇷 French", use_container_width=True):
        st.session_state.input_text = examples["French"]

with c2:
    if st.button("🇪🇸 Spanish", use_container_width=True):
        st.session_state.input_text = examples["Spanish"]
    if st.button("🇮🇳 Hindi", use_container_width=True):
        st.session_state.input_text = examples["Hindi"]

with c3:
    if st.button("🇯🇵 Japanese", use_container_width=True):
        st.session_state.input_text = examples["Japanese"]
    if st.button("🇨🇳 Chinese", use_container_width=True):
        st.session_state.input_text = examples["Chinese"]

with c4:
    if st.button("🇸🇦 Arabic", use_container_width=True):
        st.session_state.input_text = examples["Arabic"]
    if st.button("🇷🇺 Russian", use_container_width=True):
        st.session_state.input_text = examples["Russian"]

st.write("")

text = st.text_area(
    "",
    key="input_text",
    height=200,
    placeholder="Type or paste text in any supported language..."
)

# Stats
characters = len(text)
words = len(text.split())
letters = sum(ch.isalpha() for ch in text)
spaces = text.count(" ")

m1, m2, m3, m4 = st.columns(4)
for col, val, lbl in zip(
    [m1, m2, m3, m4],
    [characters, words, letters, spaces],
    ["Characters", "Words", "Letters", "Spaces"]
):
    col.markdown(f"""
    <div class='metric-pill'>
        <div class='val'>{val}</div>
        <div class='lbl'>{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

left, middle, right = st.columns([3, 2, 2])
with left:
    detect = st.button("🚀 Detect Language", use_container_width=True, type="primary")
with middle:
    clear = st.button("🗑️ Clear", use_container_width=True)
with right:
    copy = st.button("📄 Copy Text", use_container_width=True)

if clear:
    st.session_state.input_text = ""
    st.rerun()

if copy:
    st.toast("Text copied feature can be added later.")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Prediction Section
# ============================================================
if detect:

    if text.strip() == "":
        st.warning("⚠️ Please enter some text before detecting the language.")
    else:
        start_time = time.time()
        vector = cv.transform([text])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        end_time = time.time()

        inference_time = (end_time - start_time) * 1000
        confidence = np.max(probabilities) * 100

        st.write("")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Language</div>
            <div class="result-lang">🌍 {prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        c1, c2, c3 = st.columns(3)
        for col, val, lbl in zip(
            [c1, c2, c3],
            [f"{confidence:.2f}%", f"{inference_time:.2f} ms", prediction],
            ["Confidence", "Prediction Time", "Detected Language"]
        ):
            col.markdown(f"""
            <div class='metric-pill'>
                <div class='val'>{val}</div>
                <div class='lbl'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Confidence Score</div>", unsafe_allow_html=True)
        st.progress(float(confidence / 100))
        st.success(f"Model Confidence: {confidence:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        # Top 3
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📈 Top 3 Predictions</div>", unsafe_allow_html=True)

        classes = model.classes_
        top_indices = np.argsort(probabilities)[::-1][:3]
        medals = ["🥇", "🥈", "🥉"]

        for medal, idx in zip(medals, top_indices):
            language = classes[idx]
            score = probabilities[idx] * 100
            st.markdown(f"""
            <div class='rank-row'>
                <div class='rank-medal'>{medal}</div>
                <div class='rank-lang'>{language}</div>
                <div style='flex:1;'>
            """, unsafe_allow_html=True)
            st.progress(float(probabilities[idx]))
            st.markdown(f"""
                </div>
                <div class='rank-score'>{score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Probability table
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📋 Prediction Probability Table</div>", unsafe_allow_html=True)

        probability_df = pd.DataFrame({
            "Language": classes,
            "Probability (%)": np.round(probabilities * 100, 2)
        }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)

        st.dataframe(probability_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.info(f"✅ The model predicts **{prediction}** with **{confidence:.2f}% confidence**.")

        st.balloons()

# ============================================================
# About Model
# ============================================================
st.write("")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='glass-card'>
        <div class='section-title'>🤖 Machine Learning Model</div>
        <p style='color:#94a3b8; line-height:1.9; font-size:14.5px;'>
        <b>Algorithm:</b> Multinomial Naive Bayes<br>
        <b>Feature Extraction:</b> CountVectorizer<br>
        <b>Task:</b> Language Identification<br>
        <b>Input:</b> Unicode Text<br>
        <b>Output:</b> Predicted Language
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='glass-card'>
        <div class='section-title'>🚀 Project Highlights</div>
        <p style='color:#94a3b8; line-height:1.9; font-size:14.5px;'>
        ✔ Real-time Prediction<br>
        ✔ Probability Estimation<br>
        ✔ Top-3 Predictions<br>
        ✔ Supports 22 Languages<br>
        ✔ Fast Inference<br>
        ✔ Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Supported Languages
# ============================================================
st.write("")
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🌍 Supported Languages</div>", unsafe_allow_html=True)

languages_display = [
    "🇪🇪 Estonian", "🇸🇪 Swedish", "🇹🇭 Thai", "🇮🇳 Tamil", "🇳🇱 Dutch",
    "🇯🇵 Japanese", "🇹🇷 Turkish", "🏛 Latin", "🇵🇰 Urdu", "🇮🇩 Indonesian",
    "🇵🇹 Portugese", "🇫🇷 French", "🇨🇳 Chinese", "🇰🇷 Korean", "🇮🇳 Hindi",
    "🇪🇸 Spanish", "🇦🇫 Pushto", "🇮🇷 Persian", "🇷🇴 Romanian", "🇷🇺 Russian",
    "🇬🇧 English", "🇸🇦 Arabic"
]

chips_html = "".join([f"<span class='chip'>{lang}</span>" for lang in languages_display])
st.markdown(chips_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Tips
# ============================================================
st.write("")
st.markdown("""
<div class='glass-card'>
    <div class='section-title'>💡 Tips for Better Predictions</div>
    <p style='color:#94a3b8; line-height:1.9; font-size:14.5px;'>
    • Enter complete words or sentences instead of a few random characters.<br>
    • Longer text generally improves prediction confidence.<br>
    • The model supports Unicode text.<br>
    • Mixed-language sentences may reduce prediction confidence.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Developer Section
# ============================================================
st.write("")
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>👨‍💻 Developer</div>", unsafe_allow_html=True)

c1, c2 = st.columns([1, 2])

with c1:
    st.image("https://avatars.githubusercontent.com/u/9919?s=200&v=4", width=140)

with c2:
    st.markdown("""
    <p style='color:#e2e8f0; font-size:18px; font-weight:700; margin-bottom:2px;'>Bhaskar Pal</p>
    <p style='color:#64748b; font-size:13.5px; margin-top:0;'>Machine Learning · NLP · Generative AI · Data Analytics</p>
    <p style='color:#94a3b8; font-size:14.5px; line-height:1.8;'>
    This project demonstrates a multilingual language detection system built using
    <b>CountVectorizer</b> and <b>Multinomial Naive Bayes</b>. The application predicts
    one of <b>22 languages</b> from user-provided text in real time.
    </p>
    """, unsafe_allow_html=True)

    st.link_button("🌐 Portfolio", "https://bhaskarpal1707.github.io/portfolio/")
    st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/bhaskar-pal-2k02/")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Tech Stack
# ============================================================
st.write("")
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🛠 Tech Stack</div>", unsafe_allow_html=True)

t1, t2, t3, t4 = st.columns(4)
for col, val, lbl in zip(
    [t1, t2, t3, t4],
    ["Python", "Streamlit", "Naive Bayes", "CountVectorizer"],
    ["Language", "Framework", "ML Model", "Vectorizer"]
):
    col.markdown(f"""
    <div class='metric-pill'>
        <div class='val' style='font-size:18px;'>{val}</div>
        <div class='lbl'>{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown("""
<div class='footer-box'>
    <h3 style='color:#cbd5e1; font-family: Sora, sans-serif;'>🌐 LinguaAI Language Detection</h3>
    Built with ❤️ using Python • Streamlit • Scikit-learn • CountVectorizer • Multinomial Naive Bayes
    <br><br>
    Developed by <b>Bhaskar Pal</b>
    <br><br>
    © 2026 All Rights Reserved
</div>
""", unsafe_allow_html=True)