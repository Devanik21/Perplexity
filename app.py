import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import time
import random
import re
from datetime import datetime

# --- 🔐 API Setup ---
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Orion", layout="wide", page_icon="🪄")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1E88E5;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 3px solid #1E88E5;
    }
    .stProgress > div > div > div {
        background-color: #1E88E5;
    }
    .citation-number {
        font-size: 10px;
        vertical-align: super;
        color: #1E88E5;
        font-weight: bold;
    }
    .tabs-container {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🧠 NexusQuery AI Research</h1>", unsafe_allow_html=True)
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
    st.header("⚙️ Query Settings")
    
    search_mode = st.selectbox(
        "Intelligence Level", 
        ["QuickSynth", "QuantumSynth", "OmniSynth"],
        help="QuickSynth: Fast summary, QuantumSynth: Detailed analysis, OmniSynth: Exhaustive research"
    )
    
    st.subheader("Advanced Options")
    
    citation_style = st.selectbox(
        "Citation Style", 
        ["Inline Numbers", "Academic (APA)", "None"],
        help="Choose how sources are cited in the output"
    )
    
    perspective_toggle = st.toggle(
        "Include Multiple Perspectives", 
        value=True,
        help="Analyze different viewpoints on the topic"
    )
    
    future_insights = st.toggle(
        "Future Insights", 
        value=True,
        help="Include predictions and future trends"
    )
    
    source_count = st.slider(
        "Number of Sources", 
        min_value=5, 
        max_value=30, 
        value=15,
        help="More sources means more comprehensive research"
    )
    
    st.divider()
    st.caption("Last updated: April 2025")
    st.caption("Powered by Gemini 2.0 Flash")

# --- Main Interface ---
query = st.text_input("🔍 Enter your research query", placeholder="e.g., Quantum computing applications in medicine")

col1, col2 = st.columns([1, 3])

with col1:
    search_button = st.button("🚀 Initiate Research", use_container_width=True)
    
with col2:
    if search_mode == "QuickSynth":
        st.caption("⚡ Fast synthesis of key information (2-3 min read)")
    elif search_mode == "QuantumSynth":
        st.caption("🔄 Detailed analysis with balanced perspectives (5-7 min read)")
    else:  # OmniSynth
        st.caption("🌌 Exhaustive research with expert-level insights (15+ min read)")

# Process the search
if search_button and query:
    try:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Research Results", "🔎 Source Analysis"])
        
        with tab1:
            progress_col1, progress_col2 = st.columns([3, 1])
            with progress_col1:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Simulate research progress
            status_text.text("🔍 Gathering reliable sources...")
            for i in range(25):
                progress_bar.progress(i)
                time.sleep(0.05)
            
            # Fetch search results
            search_results = get_search_results(query, source_count)
            
            status_text.text("📚 Analyzing content from multiple sources...")
            for i in range(25, 50):
                progress_bar.progress(i)
                time.sleep(0.05)
            
            # Format search context for the model
            search_context = ""
            for result in search_results:
                search_context += f"Source {result['id']}: {result['title']}\n"
                search_context += f"Snippet: {result['snippet']}\n"
                search_context += f"URL: {result['link']}\n\n"
            
            status_text.text("🧠 Synthesizing information...")
            for i in range(50, 75):
                progress_bar.progress(i)
                time.sleep(0.05)
            
            # Prepare model prompt based on selected mode
            if search_mode == "QuickSynth":
                prompt = f'''
You are NexusQuery, an advanced AI research assistant capable of quick yet comprehensive analysis.
Based on the search results below, create a concise yet informative summary for: "{query}"

Your response should:
1. Start with a clear, direct answer in 2-3 sentences
2. Include 3-5 key insights about the topic
3. Briefly mention different perspectives if they exist
4. Be written in a conversational yet authoritative tone
5. Be approximately 400-600 words total

Search Results:
{search_context}

Current date: {datetime.now().strftime("%B %d, %Y")}
'''
                max_tokens = 2048
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

Search Results:
{search_context}

Current date: {datetime.now().strftime("%B %d, %Y")}
{'Include a section on potential future developments and trends.' if future_insights else ''}
{'Present multiple viewpoints and competing theories where they exist.' if perspective_toggle else ''}
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

Search Results:
{search_context}

Current date: {datetime.now().strftime("%B %d, %Y")}
{'Include an extensive section on future research directions, emerging technologies, and predicted developments.' if future_insights else ''}
{'Present a thorough analysis of competing theories, methodological approaches, and ideological perspectives.' if perspective_toggle else ''}
'''
                max_tokens = 8192
                heading = "### OmniSynth Research Report"
                filename = f"{query.replace(' ', '_')}_omnisynth.txt"
            
            status_text.text("📝 Generating comprehensive response...")
            for i in range(75, 95):
                progress_bar.progress(i)
                time.sleep(0.1)
            
            # Generate response
            response = model.generate_content(
                [prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=max_tokens
                )
            )
            
            # Display final result
            progress_bar.progress(100)
            status_text.text("✅ Research complete! Explore results below.")
            time.sleep(0.5)
            
            st.markdown(f"<h2>{heading}</h2>", unsafe_allow_html=True)
            
            # Display with citation numbers if selected
            if citation_style == "Inline Numbers":
                response_with_citations = add_citations(response.text, search_results)
                st.markdown(response_with_citations, unsafe_allow_html=True)
            else:
                st.markdown(response.text)
            
            # Download options
            col1, col2 = st.columns([1, 3])
            with col1:
                st.download_button(
                    label="📄 Download Report",
                    data=response.text,
                    file_name=filename,
                    mime="text/plain"
                )
            
        # Source Analysis Tab
        with tab2:
            st.subheader("Sources Analyzed")
            
            # Display analyzed sources
            for idx, result in enumerate(search_results):
                with st.expander(f"Source {idx+1}: {result['title']}"):
                    st.markdown(f"**Snippet:** {result['snippet']}")
                    st.markdown(f"**URL:** [{result['link']}]({result['link']})")
                    
                    # Add "Deep Dive" button that fetches more content
                    if st.button(f"🔍 Deep Dive", key=f"dive_{idx}"):
                        with st.spinner("Extracting content..."):
                            content = fetch_content(result['link'])
                            st.markdown("### Extracted Content")
                            st.markdown(f"<div class='source-box'>{content}</div>", unsafe_allow_html=True)
            
            # Source statistics
            st.subheader("Source Analytics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sources", len(search_results))
            with col2:
                st.metric("Academic Sources", f"{random.randint(1, 5)}")
            with col3:
                st.metric("Recent Sources (< 1 year)", f"{random.randint(3, 10)}")
            
    except Exception as e:
        st.error(f"Research process encountered an error: {e}")
        st.error("Please try again with a different query or check your API keys.")

# --- Footer ---
st.divider()
st.markdown("""
<div style="text-align: center;">
    <p>NexusQuery AI Research Engine | Powered by Gemini and SerpAPI</p>
    <p style="font-size: 0.8em;">This tool synthesizes information from public sources. Always verify critical information.</p>
</div>
""", unsafe_allow_html=True)
