# SmartChoice Career Advisor - Optimized Streamlit Dashboard (Fixed v5)
import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore")

# 🧠 CONFIG
st.set_page_config(page_title="SmartChoice Career Advisor", page_icon="🎯", layout="wide")

DATA_DIR = r"D:\RS\SmartChoice_Career_Advisor\data\processed"

# ⚡ CACHE FUNCTIONS
@st.cache_data
def load_data():
    occ_df = pd.read_csv(os.path.join(DATA_DIR, "Master_Eligible_Jobs.csv"))
    skills_df = pd.read_csv(os.path.join(DATA_DIR, "Skills.csv"))
    return occ_df, skills_df

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# 🧩 LOAD DATA + MODEL
occupation_df, skills_df = load_data()
model = load_model()

# ⚡ PRECOMPUTE / LOAD JOB EMBEDDINGS
embeddings_file = os.path.join(DATA_DIR, "job_embeddings.npy")
if os.path.exists(embeddings_file):
    job_embeddings = np.load(embeddings_file)
else:
    st.info("Precomputing embeddings for the first time. This may take a few minutes...")
    job_embeddings = model.encode(occupation_df['Skills'].astype(str).tolist(), normalize_embeddings=True)
    np.save(embeddings_file, job_embeddings)
    st.success("Job embeddings computed and saved for future sessions.")

# 🎯 APP HEADER
st.markdown("""
    <h1 style='text-align:center; color:#0073e6;'>SmartChoice Career Advisor</h1>
    <p style='text-align:center;'>Empowering women returning to work with AI-driven, inclusive career insights.</p>
""", unsafe_allow_html=True)

# 🧭 SIDEBAR NAVIGATION
st.sidebar.title("🔍 Navigation Menu")
page = st.sidebar.radio("Select Section", ["🏠 Home", "💼 Career Match"])

# 🏠 HOME PAGE
if page == "🏠 Home":
    st.subheader("Welcome to SmartChoice Career Advisor! 👋")
    st.markdown("""
    This AI-powered platform helps **women relaunch their careers** by aligning 
    their **skills, education, and experience** to relevant occupations.
    
    The system uses a **hybrid recommendation model** combining **TF-IDF** and **Semantic Similarity**
    for meaningful, inclusive, and context-aware career suggestions.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=300)
    st.info("Navigate to 'Career Match' to get your personalized recommendations!")

# 💼 CAREER MATCH PAGE
elif page == "💼 Career Match":
    with st.form("career_form"):
        st.write("### 👩‍💻 Tell us about yourself")

        col1, col2 = st.columns(2)
        with col1:
            user_skills = st.text_input("Enter your Skills", placeholder="e.g., Python, Excel, Communication", key="skills_input")
            user_roles = st.text_area("Previous Roles", placeholder="e.g., Data Analyst, Project Manager", key="roles_input")

        with col2:
            edu = st.selectbox("Highest Education Level",
                               ["High School", "Associate", "Bachelor", "Master", "Doctorate"], key="edu_select")
            exp = st.slider("Years of Work Experience", 0, 25, 3, key="exp_slider")
            remote = st.checkbox("Prefer Remote Work?", key="remote_check")
            part_time = st.checkbox("Prefer Part-time Roles?", key="parttime_check")

        submitted = st.form_submit_button("🚀 Find My Matches")

    # Define top_jobs outside the form to avoid reference errors
    top_jobs = pd.DataFrame()

    if submitted and user_skills.strip():
        with st.spinner("Analyzing your profile and finding best matches..."):
            # --- TF-IDF Model ---
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(occupation_df['Skills'].astype(str))
            user_vector = tfidf.transform([user_skills])
            tfidf_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

            # --- Semantic Similarity using precomputed embeddings ---
            user_embedding = model.encode([user_skills], normalize_embeddings=True)
            semantic_scores = cosine_similarity(user_embedding, job_embeddings).flatten()

            # --- Combine both scores ---
            weight = st.slider("⚖️ Adjust Weight (TF-IDF ↔ Semantic)", 0.0, 1.0, 0.5, key="weight_slider")
            occupation_df['Final Score'] = weight * tfidf_scores + (1 - weight) * semantic_scores

            # --- Women-specific prioritization ---
        def is_women_friendly(text):
             text = str(text).lower()
             keywords = ['remote', 'flexible', 'work from home', 'part-time', 'inclusive', 'hybrid']
             return any(word in text for word in keywords)
        # Ensure each run starts fresh
        occupation_df = occupation_df.copy()
        occupation_df['Women_Friendly'] = occupation_df['Description'].apply(is_women_friendly)
        # --- Women-specific prioritization (balanced) ---
        if remote or part_time:
             mask = (occupation_df['Women_Friendly']) & (occupation_df['Final Score'] > 0.20)
             occupation_df.loc[mask, 'Final Score'] *= 1.05  # gentle +5% multiplier
             # --- Final top results ---
             top_jobs = (
                     occupation_df.sort_values(by='Final Score', ascending=False)
                             .reset_index(drop=True)
                                     .head(10)
                                     )
             st.success("✅ Top Career Recommendations Found!")
             st.dataframe(top_jobs[['Title', 'Skills', 'Final Score']], use_container_width=True)
             # --- Job Insights Section ---
             st.markdown("#### 💼 Detailed Job Insights")       
        for i, row in top_jobs.iterrows():
            badge = "💖 *Women-Friendly Role*" if row.get('Women_Friendly', False) else ""
            with st.expander(f"{row['Title']} {badge}", expanded=False):
                skills_list = [s.strip() for s in str(row['Skills']).split(',') if s.strip()]
                unique_skills = sorted(set(skills_list), key=skills_list.index)

                st.markdown("**Required Skills:**")
                st.markdown(", ".join(unique_skills))

                if badge:
                    st.markdown(
                        "<span style='color:deeppink;'>✨ This role supports flexible or remote options ideal for women returning to work.</span>",
                        unsafe_allow_html=True
                    )

                st.progress(float(row['Final Score']))
