import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import time
import random
import re
from datetime import datetime
import os

# PDF Generation - you may need to install these:
# pip install markdown2 weasyprint
try:
    import markdown2
    from weasyprint import HTML, CSS
except ImportError:
    st.error("PDF generation libraries not found. Please run: pip install markdown2 weasyprint")

import base64

# --- 🔐 API Setup ---
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# --- Dark/Light Mode Config ---
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

# Set theme based on session state
light_theme = """
<style>
    .main-header { color: #1E88E5; }
    .source-box {
        background-color: #f0f2f6;
        border-left: 3px solid #1E88E5;
    }
    .citation-number { color: #1E88E5; }
    body { background-color: #ffffff; color: #333333; }
</style>
"""

dark_theme = """
<style>
    .main-header { color: #64B5F6; }
    .source-box {
        background-color: #2a2a2a;
        border-left: 3px solid #64B5F6;
    }
    .citation-number { color: #64B5F6; }
    body { background-color: #121212; color: #e0e0e0; }
    .stTextInput input, .stSelectbox div div, .stSlider div div {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
    }
</style>
"""

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Orion", layout="wide", page_icon="🪄")

# --- START: BACKGROUND IMAGE & TRANSPARENCY ---
def set_app_background(image_file):
    """Sets the background of the Streamlit app to a local image file."""
    if not os.path.exists(image_file):
        # Fail silently or show a warning if you prefer, so the app still runs
        st.warning(f"⚠️ Background image not found: '{image_file}'. Using default theme.")
        return

    with open(image_file, "rb") as f:
        img_bytes = f.read()

    base64_img = base64.b64encode(img_bytes).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

set_app_background("Gemini_Generated_Image_6zf6sd6zf6sd6zf6 (1) (1).jpeg")

# --- CUSTOM CSS: TRANSPARENT SIDEBAR & HEADER ---
st.markdown("""
<style>
    /* Make the sidebar and header transparent */
    [data-testid="stSidebar"], [data-testid="stHeader"] {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)
# --- END: BACKGROUND IMAGE & TRANSPARENCY ---

# Inject base CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .source-box {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div {
        background-color: #1E88E5;
    }
    .citation-number {
        font-size: 10px;
        vertical-align: super;
        font-weight: bold;
    }
    .tabs-container {
        margin-bottom: 20px;
    }
    .theme-toggle {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Inject theme-specific CSS
if st.session_state.theme == "light":
    st.markdown(light_theme, unsafe_allow_html=True)
else:
    st.markdown(dark_theme, unsafe_allow_html=True)

# --- START: PASSWORD PROTECTION ---
# Initialize session state variables for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

# Define the maximum number of allowed attempts
MAX_ATTEMPTS = 3

try:
    # The password should be set in your Streamlit secrets
    # e.g., in .streamlit/secrets.toml
    # research_app_password = "your_secret_password"
    correct_password = st.secrets["research_app_password"]
except KeyError:
    st.error("`research_app_password` not found in secrets.toml. Please set it to run the app.")
    st.stop()

# If not authenticated, show the login screen.
if not st.session_state.authenticated:
    
    # Check if the user is locked out
    if st.session_state.login_attempts >= MAX_ATTEMPTS:
        st.error("🚫 **Access Blocked**")
        st.warning("Too many incorrect password attempts. Please close and reopen the app to try again.")
        st.stop()

    # Display the login form
    st.markdown("<h2 class='main-header'>Login Required</h2>", unsafe_allow_html=True)
    
    password = st.text_input(
        "Enter Password",
        type="password",
        key="password_input_field",
        label_visibility="collapsed",
        placeholder="Enter password to unlock"
    )
    
    if st.button("Enter", use_container_width=True):
        if password == correct_password:
            st.session_state.authenticated = True
            st.session_state.login_attempts = 0 # Reset on success
            st.rerun()
        else:
            st.session_state.login_attempts += 1
            attempts_left = MAX_ATTEMPTS - st.session_state.login_attempts
            st.error(f"Incorrect password. You have {attempts_left} attempt(s) left.")
            time.sleep(1)
            st.rerun()

    # Stop the app from running further if not authenticated
    st.stop()
# --- END: PASSWORD PROTECTION ---

st.markdown("<h1 class='main-header'> 🌌 Deep Research Engine</h1>", unsafe_allow_html=True)
st.markdown("Unleash knowledge synthesis with AI-powered research that delivers comprehensive insights with real-time data integration.")

# --- Helper Functions ---
def get_search_results(query, num_results=20):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "engine": "google",
        "api_key": st.secrets["SERPAPI_KEY"],
        "num": num_results
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    results = []
    for idx, result in enumerate(data.get("organic_results", [])[:num_results]):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        
        results.append({
            "id": idx + 1,
            "title": title,
            "snippet": snippet,
            "link": link
        })
    
    return results

def fetch_content(url):
    """Fetch and extract main content from a webpage"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script, style elements and comments
        for element in soup(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()
            
        # Get text and clean it
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        
        # Trim to avoid token limits (first 1500 chars should have the important content)
        if len(text) > 1500:
            text = text[:1500] + "..."
            
        return text
    except Exception as e:
        return f"Error fetching content: {str(e)[:100]}"

def simulate_typing(text, container):
    """Simulate typing for dynamic content appearance"""
    message_placeholder = container.empty()
    full_text = ""
    
    for i in range(len(text)):
        full_text += text[i]
        message_placeholder.markdown(full_text + "▌")
        time.sleep(0.005)  # Adjust speed as needed
    
    message_placeholder.markdown(full_text)
    return full_text

def add_citations(text, sources):
    """Add citation numbers to text based on source presence"""
    for idx, source in enumerate(sources):
        # Look for source title keywords in text
        title_words = source["title"].split()
        significant_words = [word.lower() for word in title_words if len(word) > 4]
        
        for word in significant_words:
            if word.lower() in text.lower() and f'<span class="citation-number">[{idx+1}]</span>' not in text:
                text = text.replace(word, f'{word}<span class="citation-number">[{idx+1}]</span>', 1)
    
    return text

# --- Sidebar for Settings ---
with st.sidebar:
    # Theme Toggle
    theme_col1, theme_col2 = st.columns([1, 4])
    with theme_col1:
        theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
        st.button(theme_icon, on_click=toggle_theme, key="theme_toggle")
    with theme_col2:
        st.write(f"{'Dark' if st.session_state.theme == 'dark' else 'Light'} Mode")
    
    st.divider()
    st.header("⚙️ Query Settings")
    
    search_mode = st.selectbox(
        "Intelligence Level", 
        ["QuickSynth", "QuantumSynth", "OmniSynth"],
        help="QuickSynth: Fast summary, QuantumSynth: Detailed analysis, OmniSynth: Exhaustive research"
    )
    
    st.subheader("Advanced Options")
    
    # Original features
    citation_style = st.selectbox(
        "Citation Style", 
        ["Inline Numbers", "Academic (APA)", "None"], # Default to "Inline Numbers"
        help="Inline Numbers: Adds [1] style citations. Academic (APA): AI attempts APA (experimental). None: No specific style."
    )
    st.subheader("Future Updates(beta)")
    st.markdown("---")
    perspective_toggle = st.toggle(
        "Include Multiple Perspectives", 
        value=True,
        help="Analyze different viewpoints on the topic, citing key proponents or evidence if found."
    )
    
    # New features
    future_insights = st.toggle(
        "Future Insights", 
        value=True,
        help="Include a dedicated section on predictions, potential future developments, and emerging trends."
    )
    
    data_visualization = st.toggle(
        "Data Visualization", 
        value=False, # Defaulting to False as per your input for refinement
        help="If enabled, AI will suggest 1-2 specific charts/graphs for key data/trends (does not generate images)."
    )
    
    executive_summary = st.toggle(
        "Executive Summary", 
        value=True,
        help="Add a concise executive summary (or abstract for OmniSynth) at the beginning of the report."
    )
    
    historical_context = st.toggle(
        "Historical Context", 
        value=False,
        help="If enabled, include relevant historical background and the evolution of the topic."
    )
    
    expert_quotes = st.toggle(
        "Expert Quotes", 
        value=False,
        help="If enabled, incorporate notable quotes from domain experts when found in sources."
    )
    
    source_count = st.slider(
        "Number of Sources", 
        min_value=5, 
        max_value=100, 
        value=15,
        help="More sources means more comprehensive research"
    )
    
    st.divider()
    st.caption("Last updated: April 2025")
    st.caption("Powered by Gemini 2.5 Flash")

# --- Main Interface ---
query = st.text_input("🔍 Enter your research query", placeholder="e.g., Quantum computing applications in medicine")

col1, col2 = st.columns([1, 3])

with col1:
    search_button = st.button(" Initiate Research", use_container_width=True)
    
with col2:
    if search_mode == "QuickSynth":
        st.caption("⚡ Fast synthesis of key information (2-3 min read)")
    elif search_mode == "QuantumSynth":
        st.caption("🔄 Detailed analysis with balanced perspectives (5-7 min read)")
    else:  # OmniSynth
        st.caption("🌌 Exhaustive research with expert-level insights (15+ min read)")

# Initialize session state for storing results
if "research_complete" not in st.session_state:
    st.session_state.research_complete = False
if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "report_heading" not in st.session_state:
    st.session_state.report_heading = ""
if "report_filename" not in st.session_state:
    st.session_state.report_filename = ""

# Process the search
if search_button and query:
    st.session_state.research_complete = False # Reset on new search
    with st.spinner("Initiating research... Please wait."):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔍 Gathering reliable sources...")
            search_results = get_search_results(query, source_count)
            st.session_state.search_results = search_results # Save sources
            progress_bar.progress(25)

            status_text.text("📚 Analyzing content from multiple sources...")
            search_context = ""
            for result in search_results:
                search_context += f"Source {result['id']}: {result['title']}\n"
                search_context += f"Snippet: {result['snippet']}\n"
                search_context += f"URL: {result['link']}\n\n"
            progress_bar.progress(50)

            status_text.text("🧠 Synthesizing information...")
            # (Code for preparing prompt is unchanged, so it's omitted for brevity)
            # ...
            # Prepare model prompt based on selected mode and features
            feature_instructions = []
            if future_insights:
                feature_instructions.append("Include a dedicated section discussing potential future developments, predictions, and emerging trends related to the topic.")
            if data_visualization:
                feature_instructions.append("If the analysis reveals quantifiable data, trends, or comparisons, suggest 1-2 specific charts or graphs (e.g., bar chart, line graph, pie chart) that could visualize these insights. Clearly state what data from the provided search context would be used for each axis or segment of the suggested visualization.")
            if historical_context:
                feature_instructions.append("Include a section detailing the relevant historical background and the evolution of the topic.")
            if expert_quotes:
                feature_instructions.append("Where relevant and available in the provided sources, integrate 1-2 notable quotes or direct insights from recognized experts in the field. Attribute them clearly.")
            if perspective_toggle:
                feature_instructions.append("If multiple significant viewpoints or competing theories are evident from the search results, present them clearly. For each, briefly mention its basis or key proponents as found in the search results.")

            # Executive Summary instruction (conditional on search_mode for specific wording)
            if executive_summary:
                if search_mode == "QuickSynth":
                    feature_instructions.append("Start with a brief 'Executive Summary' (2-3 sentences) that directly answers the query and highlights key takeaways.")
                elif search_mode == "QuantumSynth":
                    feature_instructions.append("Ensure the report begins with a dedicated 'Executive Summary' (approx. 100 words) highlighting key findings and conclusions.")
                else:  # OmniSynth
                    feature_instructions.append("Ensure the report begins with a comprehensive 'Abstract' (approx. 150-200 words) summarizing the purpose, methods, key findings, and conclusions.")

            # Citation Style instruction
            if citation_style == "Academic (APA)":
                feature_instructions.append("If citing specific information from sources, attempt to use APA-style in-text citations. If compiling a reference list is feasible from snippets, attempt an APA-style list at the end. This is a best-effort approach.")

            features_text = "\n".join(feature_instructions) # Use newline for better clarity in the prompt
            if search_mode == "QuickSynth":
                prompt = f'''
You are NexusQuery, an advanced AI research assistant capable of quick yet comprehensive analysis.
Based on the search results below, create a concise yet informative summary for: "{query}"

Your response should:
1. Start with a clear, direct answer in 2-3 sentences
2. Include 3-5 key insights about the topic (unless an Executive Summary is requested, which should cover this)
3. Briefly mention different perspectives if they exist
4. Be written in a conversational yet authoritative tone
5. Be approximately 100-500 words total
6. Crucially, ensure your response is complete and does not end abruptly or get cut off
{features_text}

Search Results:
{search_context}

Current date: {datetime.now().strftime("%B %d, %Y")}
'''
                max_tokens = 4096
                heading = "### QuickSynth Results"
                filename = f"{query.replace(' ', '_')}_quicksynth.txt"
                
            elif search_mode == "QuantumSynth":
                prompt = f'''
You are NexusQuery, an advanced AI research assistant with the ability to synthesize complex information.
Based on the search results below, create a detailed analytical report on: "{query}"

Your response should:
1. Begin with an executive summary (about 100 words)
2. Provide an in-depth explanation with clear structure using H2 and H3 headings
3. Include real-world examples and case studies where relevant
4. Present multiple perspectives on controversial aspects
5. Discuss limitations and criticisms of current approaches
6. End with implications and forward-looking insights
7. Use a professional yet accessible tone
8. Be approximately 1500-2000 words
9.Crucially, ensure your response is complete and does not end abruptly or get cut off

{features_text}

Search Results:
{search_context}

Current date: {datetime.now().strftime("%B %d, %Y")}
'''
                max_tokens = 4096
                heading = "### QuantumSynth Analysis"
                filename = f"{query.replace(' ', '_')}_quantumsynth.txt"
                
            else:  # OmniSynth
                prompt = f'''
You are NexusQuery, the most advanced AI research assistant available, capable of producing academic-grade comprehensive analysis.
Based on the search results below, create an exhaustive research report on: "{query}"

Your response should:
1. Begin with an abstract summarizing the entire report (150-200 words)
2. Include a comprehensive table of contents with clear hierarchical structure
3. Provide extensive detail on all relevant aspects of the topic
4. Incorporate multiple theoretical frameworks and methodologies
5. Critically evaluate evidence, identifying strengths and weaknesses
6. Compare and contrast competing perspectives with nuanced analysis
7. Discuss historical development, current state, and future trajectories
8. Use an academic tone with precise terminology
9. Be organized with clear sections, subsections, and consistent formatting
10. Be approximately 4000-5000 words, resembling a scholarly article or white paper
11.Crucially, ensure your response is complete and does not end abruptly or get cut off

{features_text}

Search Results:
{search_context}

Current date: {datetime.now().strftime("%B %d, %Y")}
'''
                max_tokens = 8192
                st.session_state.report_heading = "### OmniSynth Research Report"
                st.session_state.report_filename = f"{query.replace(' ', '_')}_omnisynth.txt"

            progress_bar.progress(75)
            status_text.text("📝 Generating comprehensive response...")

            # Generate response
            response = model.generate_content(
                [prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=max_tokens
                )
            )

            progress_bar.progress(100)
            status_text.text("✅ Research complete! Explore results below.")
            time.sleep(0.5)
            status_text.empty() # Clear status text
            progress_bar.empty() # Clear progress bar

            if response.candidates and response.text:
                st.session_state.generated_text = response.text
                st.session_state.research_complete = True
            elif response.candidates and not response.text:
                st.warning("The AI model generated a response, but it contained no text content. This might be due to safety filters or an issue with the prompt.")
                st.session_state.research_complete = False
            else: # No candidates, likely an error during generation
                st.error("The AI model did not return a valid response. Please check the logs or try again.")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    st.error(f"Prompt Feedback: {response.prompt_feedback}")
                st.session_state.research_complete = False

        except Exception as e:
            st.error(f"Research process encountered an error: {e}")
            st.error("Please try again with a different query or check your API keys.")
            st.session_state.research_complete = False

# Display results if they exist in session state
if st.session_state.research_complete:
    tab1, tab2 = st.tabs(["📊 Research Results", "🔎 Source Analysis"])

    with tab1:
        st.markdown(f"<h2>{st.session_state.report_heading}</h2>", unsafe_allow_html=True)

        # Apply citations if needed
        if citation_style == "Inline Numbers":
            response_with_citations = add_citations(st.session_state.generated_text, st.session_state.search_results)
            st.markdown(response_with_citations, unsafe_allow_html=True)
        else:
            st.markdown(st.session_state.generated_text)

        # --- Download Buttons ---
        st.divider()
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📄 Download as Text (.txt)",
                data=st.session_state.generated_text,
                file_name=st.session_state.report_filename,
                mime="text/plain",
                use_container_width=True
            )
        with dl_col2:
            pdf_filename = st.session_state.report_filename.replace(".txt", ".pdf")
            st.download_button(
                label="📄 Download as PDF",
                data=create_pdf_report(st.session_state.generated_text, st.session_state.theme),
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True
            )

    with tab2:
        st.subheader("Sources Analyzed")
        for idx, result in enumerate(st.session_state.search_results):
            with st.expander(f"Source {idx+1}: {result['title']}"):
                st.markdown(f"**Snippet:** {result['snippet']}")
                st.markdown(f"**URL:** [{result['link']}]({result['link']})")
                if st.button(f"🔍 Deep Dive", key=f"dive_{idx}"):
                    with st.spinner("Extracting content..."):
                        content = fetch_content(result['link'])
                        st.markdown("### Extracted Content")
                        st.markdown(f"<div class='source-box'>{content}</div>", unsafe_allow_html=True)

# --- Footer ---
st.divider()
st.markdown("""
<div style="text-align: center;">
    <p>NexusQuery AI Research Engine | Powered by Gemini and SerpAPI</p>
    <p style="font-size: 0.8em;">This tool synthesizes information from public sources. Always verify critical information.</p>
</div>
""", unsafe_allow_html=True)
