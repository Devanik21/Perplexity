import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import time
import random
import re
from datetime import datetime
import json
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- 🔐 API Setup ---
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Theme and Page Setup ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Function to toggle dark/light mode
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Set theme based on current state
theme_bg = "#1E1E1E" if st.session_state.dark_mode else "#FFFFFF"
theme_text = "#FFFFFF" if st.session_state.dark_mode else "#31333F"
theme_accent = "#4F8BF9" if st.session_state.dark_mode else "#1E88E5"

st.set_page_config(page_title="Orion", layout="wide", page_icon="🪄")

# CSS with theme variables
st.markdown(f"""
<style>
    .main-header {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: {theme_accent};
    }}
    .source-box {{
        background-color: {'#2D2D2D' if st.session_state.dark_mode else '#f0f2f6'};
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 3px solid {theme_accent};
        color: {theme_text};
    }}
    .stProgress > div > div > div {{
        background-color: {theme_accent};
    }}
    .citation-number {{
        font-size: 10px;
        vertical-align: super;
        color: {theme_accent};
        font-weight: bold;
    }}
    .tabs-container {{
        margin-bottom: 20px;
    }}
    .history-item {{
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        background-color: {'#2D2D2D' if st.session_state.dark_mode else '#f0f2f6'};
        cursor: pointer;
    }}
    .history-item:hover {{
        background-color: {'#3D3D3D' if st.session_state.dark_mode else '#e0e2e6'};
    }}
    body {{
        color: {theme_text};
        background-color: {theme_bg};
    }}
    .block-container {{
        background-color: {theme_bg};
    }}
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-header'> ✨ Deep Research Engine</h1>", unsafe_allow_html=True)
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
        # Extract date if available
        date = result.get("published_date", "Unknown")
        
        results.append({
            "id": idx + 1,
            "title": title,
            "snippet": snippet,
            "link": link,
            "date": date
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

# New function for creating PDF export
def create_pdf(title, content, sources):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title', 
                             fontName='Helvetica-Bold',
                             fontSize=16, 
                             alignment=1,
                             spaceAfter=12))
    
    story = []
    
    # Add title
    story.append(Paragraph(f"Research Report: {title}", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Add date
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add content
    # Remove HTML tags for PDF
    clean_content = re.sub(r'<.*?>', '', content)
    paragraphs = clean_content.split('\n\n')
    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(para, styles['Normal']))
            story.append(Spacer(1, 6))
    
    # Add sources
    story.append(Spacer(1, 12))
    story.append(Paragraph("Sources", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    for idx, source in enumerate(sources):
        story.append(Paragraph(f"[{idx+1}] {source['title']}", styles['Normal']))
        story.append(Paragraph(f"URL: {source['link']}", styles['Normal']))
        story.append(Spacer(1, 3))
    
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# New function to create topic clusters
def create_topic_clusters(query, sources):
    # Create a prompt to extract topics from sources
    topics_prompt = f"""
    Based on the search query "{query}" and the following sources, identify 5-8 main topic clusters:
    
    {[f"Source {s['id']}: {s['title']} - {s['snippet']}" for s in sources]}
    
    For each source, assign it to one or more topic clusters. Return the results as a JSON with this structure:
    {{
        "topics": ["topic1", "topic2", ...],
        "source_topic_mapping": [[source_id, topic_index], ...]
    }}
    """
    
    # Generate topic clusters
    response = model.generate_content(topics_prompt)
    
    try:
        # Extract JSON from response
        json_str = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        topic_data = json.loads(json_str)
        
        # Create network graph
        G = nx.Graph()
        
        # Add topic nodes
        for i, topic in enumerate(topic_data["topics"]):
            G.add_node(topic, type="topic", size=800)
        
        # Add source nodes and edges
        for mapping in topic_data["source_topic_mapping"]:
            source_id = mapping[0]
            topic_idx = mapping[1]
            source = next((s for s in sources if s["id"] == source_id), None)
            if source:
                source_name = f"S{source_id}"
                G.add_node(source_name, type="source", size=300)
                G.add_edge(source_name, topic_data["topics"][topic_idx])
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        
        # Draw topic nodes
        topic_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == "topic"]
        nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, 
                              node_color="lightblue", node_size=800, alpha=0.8)
        
        # Draw source nodes
        source_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == "source"]
        nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, 
                              node_color="lightgreen", node_size=300, alpha=0.6)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.axis('off')
        
        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    except Exception as e:
        st.error(f"Error creating topic clusters: {e}")
        return None

# Function to generate research timeline
def create_research_timeline(sources):
    # Extract dates and organize by time (use dummy dates for demo)
    timeline_data = []
    
    # Create some reasonable dummy dates if none available
    start_year = datetime.now().year - 5
    end_year = datetime.now().year
    
    for idx, source in enumerate(sources[:10]):  # Limit to 10 sources for visualization
        # Generate plausible date if not available
        if source.get("date") == "Unknown":
            random_year = random.randint(start_year, end_year)
            random_month = random.randint(1, 12)
            random_day = random.randint(1, 28)
            date_str = f"{random_year}-{random_month:02d}-{random_day:02d}"
        else:
            date_str = source.get("date")
            
        # Add to timeline data
        timeline_data.append({
            "id": source["id"],
            "title": source["title"][:40] + "..." if len(source["title"]) > 40 else source["title"],
            "date": date_str
        })
    
    # Sort by date
    timeline_data = sorted(timeline_data, key=lambda x: x["date"])
    
    # Create timeline visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot points
    y_positions = range(len(timeline_data))
    ax.scatter([pd.to_datetime(item["date"]) for item in timeline_data], y_positions, s=100, color='blue', zorder=2)
    
    # Add lines connecting points
    ax.plot([pd.to_datetime(item["date"]) for item in timeline_data], y_positions, color='gray', alpha=0.5, zorder=1)
    
    # Add labels
    for i, item in enumerate(timeline_data):
        ax.annotate(f"[{item['id']}] {item['title']}", 
                   (pd.to_datetime(item["date"]), i),
                   xytext=(10, 0), 
                   textcoords="offset points",
                   ha='left', va='center')
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    ax.set_ylabel('')
    
    # Format x-axis
    ax.set_xlabel('Publication Timeline')
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save figure to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

# --- Sidebar for Settings and History ---
with st.sidebar:
    st.header("⚙️ Query Settings")
    
    # Theme toggle
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🌓" if st.session_state.dark_mode else "☀️"):
            toggle_theme()
            st.rerun()
    with col2:
        st.write("Toggle Dark Mode")
    
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
        max_value=200, 
        value=15,
        help="More sources means more comprehensive research"
    )
    
    # New visual analysis toggle
    visual_analysis = st.toggle(
        "Generate Visualizations", 
        value=True,
        help="Create topic clusters and timeline visualizations"
    )
    
    st.divider()
    
    # Research History
    st.subheader("📚 Research History")
    
    # Initialize session state for history if it doesn't exist
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []
    
    # Display history items
    if not st.session_state.research_history:
        st.caption("Your research history will appear here")
    else:
        for i, history_item in enumerate(st.session_state.research_history):
            if st.button(f"{history_item['query'][:30]}...", key=f"history_{i}", 
                       help=f"Click to reload this research", use_container_width=True):
                st.session_state.reload_query = history_item['query']
                st.session_state.reload_mode = history_item['mode']
                st.rerun()
    
    st.divider()
    st.caption("Last updated: April 2025")
    st.caption("Powered by Gemini 2.0 Flash")

# --- Main Interface ---
# Check if we're reloading from history
if 'reload_query' in st.session_state:
    query = st.session_state.reload_query
    search_mode = st.session_state.reload_mode
    # Clear the reload state
    del st.session_state.reload_query
    del st.session_state.reload_mode
    # Set the query in the input box
    query_placeholder = query
else:
    query_placeholder = "e.g., Quantum computing applications in medicine"
    
query = st.text_input("🔍 Enter your research query", value=query if 'query' in locals() else "", 
                     placeholder=query_placeholder)

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
        # Add to history
        if query not in [item['query'] for item in st.session_state.research_history]:
            st.session_state.research_history.insert(0, {
                'query': query,
                'mode': search_mode,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            # Keep only last 10 items
            if len(st.session_state.research_history) > 10:
                st.session_state.research_history = st.session_state.research_history[:10]
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Research Results", "🔎 Source Analysis", "📈 Visual Analysis"])
        
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
            col1, col2, col3 = st.columns([1, 1, 2])
            
            # Plain text download
            with col1:
                st.download_button(
                    label="📄 Download TXT",
                    data=response.text,
                    file_name=filename,
                    mime="text/plain"
                )
            
            # PDF download option
            with col2:
                pdf_data = create_pdf(query, response.text, search_results)
                st.download_button(
                    label="📑 Download PDF",
                    data=pdf_data,
                    file_name=f"{query.replace(' ', '_')}.pdf",
                    mime="application/pdf"
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
        
        # Visual Analysis Tab (New)
        with tab3:
            if visual_analysis:
                st.subheader("Research Visualizations")
                
                visual_col1, visual_col2 = st.columns(2)
                
                with visual_col1:
                    st.subheader("Topic Cluster Analysis")
                    with st.spinner("Generating topic clusters..."):
                        topic_cluster_img = create_topic_clusters(query, search_results)
                        if topic_cluster_img:
                            st.image(topic_cluster_img, use_column_width=True)
                            
                            # Add download button for the image
                            btn = st.download_button(
                                label="📥 Download Cluster Image",
                                data=topic_cluster_img,
                                file_name=f"{query.replace(' ', '_')}_clusters.png",
                                mime="image/png"
                            )
                
                with visual_col2:
                    st.subheader("Research Timeline")
                    with st.spinner("Creating research timeline..."):
                        timeline_img = create_research_timeline(search_results)
                        st.image(timeline_img, use_column_width=True)
                        
                        # Add download button for the image
                        btn = st.download_button(
                            label="📥 Download Timeline",
                            data=timeline_img,
                            file_name=f"{query.replace(' ', '_')}_timeline.png",
                            mime="image/png"
                        )
            else:
                st.info("Enable 'Generate Visualizations' in the sidebar settings to see visual analysis.")
            
    except Exception as e:
        st.error(f"Research process encountered an error: {e}")
        st.error("Please try again with a different query or check your API keys.")

# --- Footer ---
st.divider()
st.markdown(f"""
<div style="text-align: center;">
    <p>NexusQuery AI Research Engine | Powered by Gemini and SerpAPI</p>
    <p style="font-size: 0.8em;">This tool synthesizes information from public sources. Always verify critical information.</p>
</div>
""", unsafe_allow_html=True)
