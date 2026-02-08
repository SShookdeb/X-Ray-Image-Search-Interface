import streamlit as st
import os
from PIL import Image
from src.text_search import TextSearch
from src.image_search import ImageSearch

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="X-Ray Image Search",
    page_icon="🩻",
    layout="wide"
)

DATASET_DIR = "Dataset"

text_engine = TextSearch()
image_engine = ImageSearch()

# ---------------- SESSION STATE ----------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ---------------- SIDEBAR (FEATURE 1: MODE TOGGLE) ----------------
with st.sidebar:
    st.markdown("### ⚙ Interface Settings")
    st.session_state.dark_mode = st.toggle(
        "Dark mode",
        value=st.session_state.dark_mode
    )
    st.markdown("---")
    st.caption("Designed for clinical clarity and focus.")

# ---------------- THEME COLORS ----------------
if st.session_state.dark_mode:
    BG = "#0f172a"
    CARD = "#111827"
    TEXT = "#e5e7eb"
    SUB = "#9ca3af"
    ACCENT = "#60a5fa"
else:
    BG = "#f4f6f9"
    CARD = "#ffffff"
    TEXT = "#111827"
    SUB = "#6b7280"
    ACCENT = "#3730a3"

# ---------------- STYLES ----------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

body {{
    background-color: {BG};
}}

.block-container {{
    padding-block-start: 2.5rem;
    padding-block-end: 3rem;
}}

.header-title {{
    font-family: 'Playfair Display', serif;
    font-size: 34px;
    font-weight: 600;
    margin-block-end: 0.3rem;
}}

.subtitle {{
    color: {SUB};
    font-size: 15px;
    margin-block-end: 2.2rem;
}}

.card {{
    background-color: {CARD};
    padding: 1rem;
    border-radius: 14px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    margin-block-end: 1.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

.card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 14px 30px rgba(0,0,0,0.12);
}}

.badge {{
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 12px;
    background-color: rgba(96,165,250,0.15);
    color: {ACCENT};
    margin-block-start: 8px;
}}

.section-title {{
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 1rem;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='header-title'>🩻 X-Ray Image Search</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>A clinical-grade engine for searching medical X-ray images using text or visual similarity</div>",
    unsafe_allow_html=True
)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📝 Text-Based Search", "🖼 Image-Based Search"])

# ---------------- TEXT SEARCH ----------------
with tab1:
    st.markdown("<div class='section-title'>Text-Based Search</div>", unsafe_allow_html=True)

    query = st.text_input(
        "Medical keyword",
        placeholder="e.g. chest fracture, dental x-ray, spine"
    )

    if st.button("Search", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a valid medical keyword.")
        else:
            results = text_engine.search(query, top_k=6)
            cols = st.columns(3)

            for i, (_, row) in enumerate(results.iterrows()):
                img_path = None
                for root, _, files in os.walk(DATASET_DIR):
                    if row["image_name"] in files:
                        img_path = os.path.join(root, row["image_name"])
                        break

                if img_path:
                    with cols[i % 3]:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.image(Image.open(img_path), use_container_width=True)
                        st.markdown(f"**{row['image_name']}**")
                        st.markdown(
                            f"<span class='badge'>{row['category']}</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- IMAGE SEARCH ----------------
with tab2:
    st.markdown("<div class='section-title'>Image-Based Search</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload an X-ray image",
        type=["png", "jpg", "jpeg"]
    )

    # -------- FEATURE 2: RESULT CONTROLS --------
    col_a, col_b = st.columns(2)
    with col_a:
        min_similarity = st.slider(
            "Minimum similarity threshold",
            0.0, 1.0, 0.25, 0.05
        )
    with col_b:
        sort_mode = st.selectbox(
            "Sort results by",
            ["Similarity (High → Low)", "Image Name (A → Z)"]
        )

    if uploaded:
        with open("temp_query.png", "wb") as f:
            f.write(uploaded.read())

        st.image("temp_query.png", caption="Query Image", width=260)

        if st.button("Find Similar Images", use_container_width=True):
            raw_results = image_engine.search("temp_query.png", top_k=10)

            # Apply threshold
            results = [(n, s) for n, s in raw_results if s >= min_similarity]

            # Apply sorting
            if sort_mode.startswith("Image"):
                results = sorted(results, key=lambda x: x[0])
            else:
                results = sorted(results, key=lambda x: x[1], reverse=True)

            cols = st.columns(3)

            for i, (name, score) in enumerate(results[:6]):
                img_path = None
                for root, _, files in os.walk(DATASET_DIR):
                    if name in files:
                        img_path = os.path.join(root, name)
                        break

                if img_path:
                    with cols[i % 3]:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.image(Image.open(img_path), use_container_width=True)
                        st.markdown(f"**{name}**")
                        st.markdown(
                            f"<span class='badge'>Similarity: {score:.3f}</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
