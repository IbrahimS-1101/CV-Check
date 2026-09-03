import streamlit as st
from gemini_model import create_gemini_client, generate_content_with_fallback, get_response_text
from pypdf import PdfReader
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="CV Check", page_icon="✅", layout="centered")

# Auth
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif os.getenv("GEMINI_API_KEY"):
        api_key = os.getenv("GEMINI_API_KEY")
except:
    pass

# --- FOOTER ---
def show_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding-top: 20px;">
            <a href="https://buymeacoffee.com/isamir" target="_blank">
                <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 180px !important;" >
            </a>
            <p style="margin-top: 15px; color: #aaa; font-size: 0.9em;">
                This tool is 100% free. If it saved you time, a coffee is always appreciated! ☕
            </p>
            <p style="color: #999; font-size: 0.8em;">
                Made by Ibrahim Samir | <a href="https://takea5.com" target="_blank" style="color: #999; text-decoration: none;">Takea5.com</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 2. LOGIC ---
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return None

def analyze_cv(text, api_key):
    client = create_gemini_client(api_key)
    
    prompt = f"""
    You are an expert Career Coach. 
    Analyze the resume text below.
    
    ---
    🚨 STRICT LANGUAGE PROTOCOL:
    1. Identify the DOMINANT language of the resume text.
    2. If the resume is in English, your output MUST be in English.
    3. If the resume is in Arabic, your output MUST be in Arabic.
    4. Do NOT translate the output into a different language than the source text.
    5. Ignore proper nouns (names, cities) when detecting language.
    ---
    
    Resume Text:
    {text}
    
    ---
    Output Structure (Use the DOMINANT language of the resume):
    
    ## 📊 [Score Header]: [Score]/100
    
    ### ✅ [Strengths Header]
    * [List 2-3 strong points]
    
    ### ⚠️ [Weaknesses Header]
    * [List 2-3 things to fix]
    
    ### 💡 [Action Plan Header]
    * [Specific advice 1]
    * [Specific advice 2]
    
    ### ✍️ [Rewrite Header]
    Find a weak sentence and rewrite it to be stronger.
    **[Original Label]:** [The weak phrase]
    **[Better Label]:** [The professional version]
    """
    
    response, model_name = generate_content_with_fallback(client, prompt, api_key)
    return get_response_text(response), model_name

# --- 3. UI LAYOUT ---
st.title("✅ CV Check")
st.markdown("Get instant, AI-powered feedback on your resume. Private & Free.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    if api_key:
        st.success("✅ System Online")
    else:
        api_key = st.text_input("API Key", type="password")
    
    st.info("💡 Tip: A good CV focuses on **Achievements**, not just duties.")
    st.caption("Model selection: automatic discovery with fallback.")

# Main Area
uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

if uploaded_file and api_key:
    if st.button("Analyze my CV", type="primary"):
        with st.spinner("Reading document..."):
            # 1. Extract Text
            cv_text = extract_text_from_pdf(uploaded_file)
            
            if cv_text:
                # 2. Analyze (Now with Error Handling)
                with st.spinner("AI Coach is reviewing..."):
                    try:
                        analysis, model_name = analyze_cv(cv_text, api_key)
                        st.markdown("---")
                        st.markdown(analysis)
                        st.caption(f"Model used: {model_name}")
                    except Exception as e:
                        message = str(e).lower()
                        if "quota" in message or "rate" in message or "429" in message:
                            st.warning("🚦 Gemini is rate-limited right now. Please wait a moment and try again.")
                        else:
                            st.error("⚠️ Could not analyze this CV. Please try again or check your API key.")
            else:
                st.error("Could not read the PDF. Please try a text-based PDF (not a scanned image).")

# Show Footer
show_footer()
