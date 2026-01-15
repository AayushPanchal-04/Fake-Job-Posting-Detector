import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="🔍 Fake Job Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: #f5f7fb;
    }
    
    /* Card styling */
    .main-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    /* Header styling */
    .big-font {
        font-size: 3.5rem !important;
        font-weight: 800;
        color: black;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: black ;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Alert boxes */
    .alert-fraud {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
        animation: pulse 2s infinite;
    }
    
    .alert-safe {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
    }
    
    /* Stats box */
    .stats-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize NLTK
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Load models
@st.cache_resource
def load_models():
    try:
        with open('job_detector_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, vectorizer, metadata
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the training script first.")
        return None, None, None

# Text preprocessing
def preprocess_text(text):
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# Prediction function
def predict_job_posting(model, vectorizer, title, company_profile, description, requirements, benefits):
    combined_text = f"{title} {company_profile} {description} {requirements} {benefits}"
    cleaned_text = preprocess_text(combined_text)
    
    if not cleaned_text:
        return None, None
    
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    return prediction, probability

# Create gauge chart
def create_gauge_chart(probability, prediction):
    if prediction == 1:
        color = "red"
        label = "FRAUDULENT"
    else:
        color = "green"
        label = "LEGITIMATE"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b>{label}</b>", 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70}}))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "darkblue", 'family': "Arial"},
        height=300
    )
    
    return fig

# Main app
def main():
    # Load models
    model, vectorizer, metadata = load_models()
    
    if model is None:
        st.stop()
    
    # Header
    st.markdown('<p class="big-font">🔍 Fake Job Posting Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Job Posting Verification System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        st.markdown(f"**Model Type:** {metadata['best_model']}")
        st.markdown(f"**Accuracy:** {metadata['accuracy']:.2%}")
        st.markdown(f"**ROC-AUC:** {metadata['roc_auc']:.2%}")
        
        st.markdown("---")
        st.markdown("### 📈 Dataset Stats")
        st.markdown(f"**Training Samples:** {metadata['training_samples']:,}")
        st.markdown(f"**Test Samples:** {metadata['test_samples']:,}")
        st.markdown(f"**Features:** {metadata['features']:,}")
        
        st.markdown("---")
        st.markdown("### ⚠️ Warning Signs")
        st.markdown("""
        - 🚩 Vague job descriptions
        - 🚩 Unrealistic salary promises
        - 🚩 Poor grammar/spelling
        - 🚩 No company information
        - 🚩 Requests for personal info
        - 🚩 Unprofessional email
        """)
        
        st.markdown("---")
        st.markdown("### 🛡️ Stay Safe")
        st.markdown("""
        Always verify job postings through:
        - Company's official website
        - LinkedIn company page
        - Glassdoor reviews
        - Better Business Bureau
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Job Posting", "📊 Model Performance", "💡 About"])
    
    with tab1:
        st.markdown("### Enter Job Posting Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("📝 Job Title", placeholder="e.g., Senior Software Engineer")
            company_profile = st.text_area("🏢 Company Profile", 
                                          placeholder="Brief description of the company...",
                                          height=150)
            benefits = st.text_area("💰 Benefits", 
                                   placeholder="List of benefits offered...",
                                   height=150)
        
        with col2:
            description = st.text_area("📄 Job Description", 
                                      placeholder="Detailed job description...",
                                      height=200)
            requirements = st.text_area("✅ Requirements", 
                                       placeholder="Required skills and qualifications...",
                                       height=200)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Job Posting", use_container_width=True)
        
        if analyze_button:
            if not title and not description:
                st.warning("⚠️ Please provide at least a job title and description.")
            else:
                with st.spinner("🔄 Analyzing job posting..."):
                    time.sleep(1)  # Simulate processing
                    prediction, probability = predict_job_posting(
                        model, vectorizer, title, company_profile, 
                        description, requirements, benefits
                    )
                    
                    if prediction is None:
                        st.error("❌ Unable to analyze. Please provide more information.")
                    else:
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col2:
                            # Gauge chart
                            fig = create_gauge_chart(probability[1], prediction)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Result alert
                        if prediction == 1:
                            fraud_score = probability[1] * 100
                            st.markdown(f"""
                            <div class="alert-fraud">
                                ⚠️ WARNING: POTENTIALLY FRAUDULENT JOB POSTING<br>
                                <span style="font-size: 2rem;">🚨 {fraud_score:.1f}% Fraud Probability</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class="info-box">
                                <b>⚠️ Recommendation:</b> Exercise extreme caution! This job posting shows characteristics 
                                commonly associated with fraudulent listings. We strongly recommend:
                                <ul>
                                    <li>Verify the company independently</li>
                                    <li>Do not share personal information</li>
                                    <li>Avoid any requests for payment</li>
                                    <li>Research the company on trusted platforms</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            safe_score = probability[0] * 100
                            st.markdown(f"""
                            <div class="alert-safe">
                                ✅ LIKELY LEGITIMATE JOB POSTING<br>
                                <span style="font-size: 2rem;">🛡️ {safe_score:.1f}% Legitimacy Confidence</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class="info-box" style="background: linear-gradient(135deg, #d4f1f4 0%, #a3e4d7 100%); border-left-color: #27ae60;">
                                <b>✅ Analysis:</b> This job posting appears to be legitimate based on our AI model. 
                                However, always practice due diligence:
                                <ul>
                                    <li>Verify through official company channels</li>
                                    <li>Check company reviews and reputation</li>
                                    <li>Be cautious of any unusual requests</li>
                                    <li>Trust your instincts</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed probabilities
                        st.markdown("### 📈 Detailed Confidence Scores")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="stats-box">
                                <h3 style="color: #27ae60;">✅ Legitimate</h3>
                                <h2 style="color: #27ae60;">{probability[0]*100:.2f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="stats-box">
                                <h3 style="color: #e74c3c;">⚠️ Fraudulent</h3>
                                <h2 style="color: #e74c3c;">{probability[1]*100:.2f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Model Performance Metrics")
        
        # Model comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Accuracy Comparison")
            model_names = list(metadata['model_results'].keys())
            accuracies = [metadata['model_results'][m]['accuracy'] for m in model_names]
            
            fig = px.bar(
                x=model_names,
                y=accuracies,
                labels={'x': 'Model', 'y': 'Accuracy'},
                color=accuracies,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 ROC-AUC Comparison")
            roc_aucs = [metadata['model_results'][m]['roc_auc'] for m in model_names]
            
            fig = px.bar(
                x=model_names,
                y=roc_aucs,
                labels={'x': 'Model', 'y': 'ROC-AUC Score'},
                color=roc_aucs,
                color_continuous_scale='Plasma'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model details table
        st.markdown("#### 📋 Detailed Performance")
        df_results = pd.DataFrame(metadata['model_results']).T
        df_results.columns = ['Accuracy', 'ROC-AUC']
        df_results = df_results.round(4)
        st.dataframe(df_results, use_container_width=True)
    
    with tab3:
        st.markdown("### 💡 About This Application")
        
        st.markdown("""
        <div class="main-card">
            <h3>🎯 Purpose</h3>
            <p>This AI-powered application helps job seekers identify potentially fraudulent job postings 
            using advanced machine learning algorithms. It analyzes various aspects of job postings to 
            provide an assessment of their legitimacy.</p>
            
            <h3>🤖 How It Works</h3>
            <p>The system uses Natural Language Processing (NLP) and machine learning to:</p>
            <ul>
                <li>Analyze job posting text patterns</li>
                <li>Identify suspicious language and content</li>
                <li>Compare against known fraudulent posting characteristics</li>
                <li>Provide probability scores for legitimacy</li>
            </ul>
            
            <h3>📚 Dataset</h3>
            <p>Trained on the <b>Real or Fake Job Posting Prediction</b> dataset from Kaggle, 
            containing thousands of labeled job postings.</p>
            
            <h3>⚖️ Disclaimer</h3>
            <p><i>This tool provides predictions based on machine learning models and should be used as 
            one factor in your decision-making process. Always conduct thorough research and verification 
            before applying to any job posting or sharing personal information.</i></p>
            
            <h3>🔒 Privacy</h3>
            <p>All analysis is performed locally. No job posting data is stored or transmitted to 
            external servers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: black ; padding: 1rem;">
            <p>Built by Aayush Panchal</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()