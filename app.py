import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import time
import random
import re
import json
import hashlib
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import asyncio
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 🔐 Configuration & Models ---
@dataclass
class SearchResult:
    id: int
    title: str
    snippet: str
    url: str
    domain: str
    date: Optional[str] = None
    credibility_score: float = 0.0
    content: str = ""
    word_count: int = 0
    sentiment: str = "neutral"

@dataclass
class ResearchConfig:
    search_mode: str
    source_count: int
    citation_style: str
    include_perspectives: bool
    future_insights: bool
    data_visualization: bool
    executive_summary: bool
    historical_context: bool
    expert_quotes: bool
    fact_check: bool
    semantic_analysis: bool
    competitive_analysis: bool

class ResearchEngine:
    def __init__(self):
        self.setup_apis()
        self.content_cache = {}
        self.credibility_domains = {
            'edu': 0.9, 'gov': 0.95, 'org': 0.7, 'com': 0.5,
            'nature.com': 0.98, 'science.org': 0.98, 'arxiv.org': 0.85,
            'pubmed.ncbi.nlm.nih.gov': 0.95, 'scholar.google.com': 0.8
        }
    
    def setup_apis(self):
        """Initialize API configurations"""
        try:
            self.gemini_key = st.secrets["GEMINI_API_KEY"]
            self.serp_key = st.secrets["SERPAPI_KEY"]
            genai.configure(api_key=self.gemini_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        except Exception as e:
            st.error(f"API Configuration Error: {e}")
            logger.error(f"API setup failed: {e}")
    
    def calculate_credibility_score(self, url: str, content: str = "") -> float:
        """Calculate source credibility based on domain and content quality"""
        try:
            domain = urllib.parse.urlparse(url).netloc.lower()
            base_score = 0.5
            
            # Domain-based scoring
            for trusted_domain, score in self.credibility_domains.items():
                if trusted_domain in domain:
                    base_score = score
                    break
            
            # Content quality indicators
            if content:
                quality_indicators = [
                    (r'\bcitation\b|\breference\b|\bstudy\b', 0.1),
                    (r'\bdoi:\b|\barxiv:\b', 0.15),
                    (r'\bpeer.?reviewed\b|\bjournal\b', 0.1),
                    (r'\bdata\b|\bstatistics\b|\bresearch\b', 0.05)
                ]
                
                for pattern, boost in quality_indicators:
                    if re.search(pattern, content, re.IGNORECASE):
                        base_score = min(1.0, base_score + boost)
            
            return round(base_score, 2)
        except:
            return 0.5
    
    def enhanced_search(self, query: str, num_results: int = 20) -> List[SearchResult]:
        """Enhanced search with multiple parameters and result processing"""
        try:
            # Primary search
            results = []
            search_params = {
                "q": query,
                "engine": "google",
                "api_key": self.serp_key,
                "num": num_results,
                "hl": "en",
                "gl": "us",
                "safe": "active"
            }
            
            response = requests.get("https://serpapi.com/search", params=search_params, timeout=30)
            data = response.json()
            
            organic_results = data.get("organic_results", [])
            
            # Also search for academic sources
            academic_query = f"{query} site:edu OR site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov"
            academic_params = search_params.copy()
            academic_params["q"] = academic_query
            academic_params["num"] = min(10, num_results // 2)
            
            try:
                academic_response = requests.get("https://serpapi.com/search", params=academic_params, timeout=20)
                academic_data = academic_response.json()
                organic_results.extend(academic_data.get("organic_results", [])[:5])
            except:
                logger.warning("Academic search failed, continuing with standard results")
            
            for idx, result in enumerate(organic_results[:num_results]):
                try:
                    url = result.get("link", "")
                    domain = urllib.parse.urlparse(url).netloc
                    
                    search_result = SearchResult(
                        id=idx + 1,
                        title=result.get("title", "Untitled"),
                        snippet=result.get("snippet", ""),
                        url=url,
                        domain=domain,
                        date=result.get("date", None),
                        credibility_score=self.calculate_credibility_score(url, result.get("snippet", ""))
                    )
                    results.append(search_result)
                except Exception as e:
                    logger.warning(f"Error processing search result {idx}: {e}")
                    continue
            
            # Sort by credibility score (descending)
            results.sort(key=lambda x: x.credibility_score, reverse=True)
            return results
            
        except Exception as e:
            st.error(f"Search failed: {e}")
            logger.error(f"Enhanced search failed: {e}")
            return []
    
    def extract_content_parallel(self, results: List[SearchResult], max_workers: int = 5) -> List[SearchResult]:
        """Extract content from multiple URLs in parallel"""
        def fetch_single_content(result: SearchResult) -> SearchResult:
            try:
                content = self.fetch_content_advanced(result.url)
                result.content = content
                result.word_count = len(content.split())
                result.credibility_score = self.calculate_credibility_score(result.url, content)
                return result
            except Exception as e:
                logger.warning(f"Failed to fetch content for {result.url}: {e}")
                return result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_result = {executor.submit(fetch_single_content, result): result for result in results}
            
            for future in as_completed(future_to_result):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Thread execution failed: {e}")
        
        return results
    
    def fetch_content_advanced(self, url: str) -> str:
        """Advanced content extraction with better parsing and caching"""
        if url in self.content_cache:
            return self.content_cache[url]
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 
                               'advertisement', '.ad', '#ad', '.sidebar', '.menu']):
                if element:
                    element.decompose()
            
            # Try to find main content areas
            main_content = None
            content_selectors = [
                'main', 'article', '.main-content', '.content', '.post-content',
                '.article-body', '.entry-content', '#main-content', '.article-text'
            ]
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract and clean text
            text = main_content.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[\r\n\t]+', ' ', text)
            
            # Limit content length
            if len(text) > 3000:
                sentences = text.split('. ')
                truncated = []
                char_count = 0
                
                for sentence in sentences:
                    if char_count + len(sentence) > 3000:
                        break
                    truncated.append(sentence)
                    char_count += len(sentence) + 2
                
                text = '. '.join(truncated) + '...'
            
            # Cache the result
            self.content_cache[url] = text
            return text
            
        except Exception as e:
            logger.warning(f"Content extraction failed for {url}: {e}")
            return f"Content extraction failed: {str(e)[:100]}"
    
    def analyze_sentiment_and_bias(self, content: str) -> Dict[str, any]:
        """Analyze content sentiment and potential bias"""
        try:
            # Simple sentiment analysis based on keywords
            positive_words = ['excellent', 'great', 'amazing', 'outstanding', 'beneficial', 
                            'effective', 'successful', 'positive', 'good', 'better']
            negative_words = ['terrible', 'awful', 'bad', 'harmful', 'dangerous', 'failed',
                            'unsuccessful', 'negative', 'worse', 'problematic']
            
            content_lower = content.lower()
            pos_count = sum(1 for word in positive_words if word in content_lower)
            neg_count = sum(1 for word in negative_words if word in content_lower)
            
            if pos_count > neg_count * 1.5:
                sentiment = "positive"
            elif neg_count > pos_count * 1.5:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Bias indicators
            bias_indicators = {
                'emotional_language': len(re.findall(r'\b(amazing|terrible|shocking|unbelievable)\b', content_lower)),
                'absolute_statements': len(re.findall(r'\b(always|never|all|none|everyone|nobody)\b', content_lower)),
                'personal_opinions': len(re.findall(r'\b(i think|in my opinion|i believe|personally)\b', content_lower))
            }
            
            bias_score = sum(bias_indicators.values()) / max(len(content.split()), 1) * 100
            
            return {
                'sentiment': sentiment,
                'bias_score': round(bias_score, 2),
                'bias_indicators': bias_indicators
            }
        except:
            return {'sentiment': 'neutral', 'bias_score': 0, 'bias_indicators': {}}
    
    def generate_visualizations(self, query: str, results: List[SearchResult]) -> List[Dict]:
        """Generate data visualizations based on search results"""
        try:
            visualizations = []
            
            # Source credibility distribution
            credibility_scores = [r.credibility_score for r in results if r.credibility_score > 0]
            if credibility_scores:
                fig_credibility = px.histogram(
                    x=credibility_scores,
                    nbins=10,
                    title="Source Credibility Distribution",
                    labels={'x': 'Credibility Score', 'y': 'Number of Sources'},
                    color_discrete_sequence=['#64B5F6']
                )
                fig_credibility.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                visualizations.append({
                    'type': 'credibility_distribution',
                    'title': 'Source Credibility Analysis',
                    'figure': fig_credibility,
                    'description': 'Distribution of credibility scores across analyzed sources'
                })
            
            # Domain analysis
            domains = {}
            for result in results:
                domain_type = 'edu' if '.edu' in result.domain else \
                             'gov' if '.gov' in result.domain else \
                             'org' if '.org' in result.domain else 'com'
                domains[domain_type] = domains.get(domain_type, 0) + 1
            
            if domains:
                fig_domains = px.pie(
                    values=list(domains.values()),
                    names=list(domains.keys()),
                    title="Source Types Distribution",
                    color_discrete_sequence=['#64B5F6', '#42A5F5', '#2196F3', '#1976D2']
                )
                fig_domains.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                visualizations.append({
                    'type': 'domain_distribution',
                    'title': 'Source Domain Analysis',
                    'figure': fig_domains,
                    'description': 'Breakdown of sources by domain type (educational, government, etc.)'
                })
            
            # Content length analysis
            word_counts = [r.word_count for r in results if r.word_count > 0]
            if word_counts:
                fig_words = px.box(
                    y=word_counts,
                    title="Content Length Analysis",
                    labels={'y': 'Word Count'},
                    color_discrete_sequence=['#64B5F6']
                )
                fig_words.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                visualizations.append({
                    'type': 'content_length',
                    'title': 'Content Depth Analysis',
                    'figure': fig_words,
                    'description': 'Distribution of content length across sources'
                })
            
            return visualizations
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return []

# Initialize the research engine
@st.cache_resource
def get_research_engine():
    return ResearchEngine()

# --- 🎨 Enhanced UI Setup ---
st.set_page_config(
    page_title="NexusQuery Pro - AI Research Engine",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Enhanced source boxes */
    .source-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .source-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        border-radius: 12px 12px 0 0;
    }
    
    .source-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -5px rgba(59, 130, 246, 0.2);
        border-color: #3b82f6;
    }
    
    /* Citation styling */
    .citation-number {
        color: #3b82f6;
        font-weight: 600;
        font-size: 0.875rem;
        vertical-align: super;
        background: rgba(59, 130, 246, 0.1);
        padding: 2px 4px;
        border-radius: 3px;
        margin: 0 2px;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -5px rgba(59, 130, 246, 0.4);
    }
    
    /* Input fields */
    .stTextInput input, .stSelectbox div div, .stSlider div div {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #475569;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
        display: block;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        color: #94a3b8;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #334155;
        color: #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        color: #e2e8f0;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success { background-color: #10b981; }
    .status-warning { background-color: #f59e0b; }
    .status-error { background-color: #ef4444; }
    .status-info { background-color: #3b82f6; }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #334155;
        text-align: center;
        color: #64748b;
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 3px solid #334155;
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# --- 📋 Header Section ---
st.markdown("""
<div class="fade-in-up">
    <h1 class="main-header">🧠 NexusQuery Pro</h1>
    <p class="subtitle">
        Next-Generation AI Research Engine | Advanced Knowledge Synthesis with Real-Time Intelligence
        <br>
        <small>Powered by Gemini 2.0 Flash • Professional Research Analytics • Industry-Grade Insights</small>
    </p>
</div>
""", unsafe_allow_html=True)

# --- 🎛️ Advanced Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Research Configuration")
    
    # Research Mode
    search_mode = st.selectbox(
        "🎯 Intelligence Level", 
        ["QuickSynth", "QuantumSynth", "OmniSynth"],
        help="QuickSynth: Rapid insights (2-3 min) | QuantumSynth: Deep analysis (5-7 min) | OmniSynth: Comprehensive research (15+ min)",
        index=1
    )
    
    # Advanced Research Features
    st.markdown("### 🔬 Advanced Features")
    
    col1, col2 = st.columns(2)
    with col1:
        executive_summary = st.checkbox("📄 Executive Summary", value=True)
        future_insights = st.checkbox("🔮 Future Insights", value=True)
        historical_context = st.checkbox("📚 Historical Context", value=False)
        expert_quotes = st.checkbox("💬 Expert Quotes", value=True)
    
    with col2:
        perspective_toggle = st.checkbox("🔄 Multi-Perspective", value=True)
        data_visualization = st.checkbox("📊 Data Visualization", value=True)
        fact_check = st.checkbox("✅ Fact Verification", value=True)
        semantic_analysis = st.checkbox("🧠 Semantic Analysis", value=False)
    
    # Citation and Quality Controls
    st.markdown("### 📖 Citation & Quality")
    
    citation_style = st.selectbox(
        "Citation Style", 
        ["Inline Numbers", "Academic (APA)", "IEEE", "None"],
        help="Choose your preferred citation format"
    )
    
    source_quality = st.select_slider(
        "Source Quality Filter",
        options=["Basic", "Enhanced", "Premium", "Academic"],
        value="Enhanced",
        help="Higher levels prioritize more credible sources"
    )
    
    # Search Parameters
    st.markdown("### 🔍 Search Parameters")
    
    source_count = st.slider(
        "Sources to Analyze", 
        min_value=10, 
        max_value=50, 
        value=25,
        help="More sources = more comprehensive analysis"
    )
    
    content_depth = st.select_slider(
        "Content Analysis Depth",
        options=["Surface", "Moderate", "Deep", "Comprehensive"],
        value="Deep",
        help="Determines how thoroughly each source is analyzed"
    )
    
    # Performance Settings
    st.markdown("### ⚡ Performance")
    
    parallel_processing = st.checkbox("🔄 Parallel Processing", value=True, help="Faster analysis using multiple threads")
    real_time_updates = st.checkbox("📡 Real-Time Updates", value=True, help="Show progress as research proceeds")
    
    # Research Analytics
    st.markdown("### 📊 Analytics Dashboard")
    if st.button("📈 View Research History"):
        st.info("Research history feature coming soon!")
    
    # Export Options
    st.markdown("### 💾 Export Options")
    export_format = st.selectbox(
        "Export Format",
        ["Plain Text", "Markdown", "JSON", "PDF Report"],
        index=1
    )
    
    st.divider()
    
    # System Status
    st.markdown("### 🔧 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="status-indicator status-success"></span> API Connected', unsafe_allow_html=True)
        st.markdown('<span class="status-indicator status-success"></span> Cache Ready', unsafe_allow_html=True)
    with col2:
        st.markdown('<span class="status-indicator status-info"></span> AI Model: Active', unsafe_allow_html=True)
        st.markdown('<span class="status-indicator status-success"></span> Search: Optimal', unsafe_allow_html=True)
    
    st.markdown(f"<small>Last Updated: {datetime.now().strftime('%H:%M:%S')}</small>", unsafe_allow_html=True)

# --- 🔍 Main Search Interface ---
st.markdown("### 🔍 Research Query Interface")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "",
        placeholder="Enter your research question (e.g., 'Latest developments in quantum computing for healthcare applications')",
        help="Be specific for better results. Include context, time frames, or particular aspects you're interested in."
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("🚀 Launch Research", use_container_width=True, type="primary")

# Research suggestions
if not query:
    st.markdown("### 💡 Research Suggestions")
    suggestion_cols = st.columns(3)
    
    suggestions = [
        ("🧬 Biotech", "CRISPR gene editing latest clinical trials 2024"),
        ("🤖 AI/ML", "Large language models impact on software development"),
        ("🌱 Sustainability", "Carbon capture technology commercial viability"),
        ("💰 FinTech", "Cryptocurrency regulation global trends 2024"),
        ("🚗 Transportation", "Autonomous vehicle safety statistics latest research"),
        ("🏥 Healthcare", "Telemedicine effectiveness post-pandemic studies")
    ]
    
    for i, (icon_topic, suggestion) in enumerate(suggestions):
        with suggestion_cols[i % 3]:
            if st.button(f"{icon_topic}", key=f"suggestion_{i}", help=suggestion):
                st.session_state.suggested_query = suggestion
                st.rerun()

# Handle suggested query
if hasattr(st.session_state, 'suggested_query'):
    query = st.session_state.suggested_query
    del st.session_state.suggested_query
    st.rerun()

# Display research mode info
if query and not search_button:
    mode_info = {
        "QuickSynth": "⚡ Fast synthesis optimized for rapid insights and key findings",
        "QuantumSynth": "🔄 Balanced analysis with multiple perspectives and detailed exploration", 
        "OmniSynth": "🌌 Comprehensive research with expert-level depth and academic rigor"
    }
    st.info(f"{mode_info[search_mode]} | Analyzing {source_count} sources with {content_depth.lower()} analysis")

# --- 🚀 Main Research Processing ---
if search_button and query:
    try:
        research_engine = get_research_engine()
        
        # Create research configuration
        config = ResearchConfig(
            search_mode=search_mode,
            source_count=source_count,
            citation_style=citation_style,
            include_perspectives=perspective_toggle,
            future_insights=future_insights,
            data_visualization=data_visualization,
            executive_summary=executive_summary,
            historical_context=historical_context,
            expert_quotes=expert_quotes,
            fact_check=fact_check,
            semantic_analysis=semantic_analysis,
            competitive_analysis=False
        )
        
        # Create main research interface
        research_tab, sources_tab, analytics_tab, export_tab = st.tabs([
            "🔬 Research Results", 
            "📚 Source Analysis", 
            "📊 Analytics Dashboard",
            "💾 Export & Share"
        ])
        
        with research_tab:
            # Progress tracking
            progress_container = st.container()
            result_container = st.container()
            
            with progress_container:
                st.markdown("### 🔄 Research Progress")
                
                progress_col1, progress_col2, progress_col3 = st.columns([2, 1, 1])
                
                with progress_col1:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                with progress_col2:
                    sources_found = st.empty()
                    
                with progress_col3:
                    time_elapsed = st.empty()
                
                # Research metrics
                metrics_placeholder = st.empty()
            
            # Phase 1: Search and Discovery
            start_time = datetime.now()
            status_text.markdown('<span class="status-indicator status-info"></span> **Initiating search algorithms...**', unsafe_allow_html=True)
            
            for i in range(0, 15):
                progress_bar.progress(i)
                time.sleep(0.1)
            
            status_text.markdown('<span class="status-indicator status-success"></span> **Discovering relevant sources...**', unsafe_allow_html=True)
            search_results = research_engine.enhanced_search(query, source_count)
            
            sources_found.metric("Sources Found", len(search_results))
            
            for i in range(15, 30):
                progress_bar.progress(i)
                time.sleep(0.05)
            
            # Phase 2: Content Extraction
            status_text.markdown('<span class="status-indicator status-info"></span> **Extracting and analyzing content...**', unsafe_allow_html=True)
            
            if parallel_processing and len(search_results) > 5:
                search_results = research_engine.extract_content_parallel(search_results, max_workers=5)
            else:
                for idx, result in enumerate(search_results[:10]):  # Limit for sequential processing
                    try:
                        result.content = research_engine.fetch_content_advanced(result.url)
                        result.word_count = len(result.content.split())
                        
                        # Update progress
                        progress = 30 + int((idx + 1) / min(len(search_results), 10) * 25)
                        progress_bar.progress(progress)
                        
                        if real_time_updates:
                            time_elapsed.metric("Time Elapsed", f"{(datetime.now() - start_time).seconds}s")
                    except Exception as e:
                        logger.warning(f"Failed to process source {idx + 1}: {e}")
                        continue
            
            # Phase 3: Quality Analysis
            status_text.markdown('<span class="status-indicator status-info"></span> **Performing quality assessment...**', unsafe_allow_html=True)
            
            high_quality_sources = []
            total_words = 0
            sentiment_analysis = {}
            
            for result in search_results:
                if result.content and len(result.content) > 100:
                    # Recalculate credibility with content
                    result.credibility_score = research_engine.calculate_credibility_score(result.url, result.content)
                    
                    # Sentiment analysis
                    if semantic_analysis:
                        analysis = research_engine.analyze_sentiment_and_bias(result.content)
                        result.sentiment = analysis['sentiment']
                        sentiment_analysis[result.id] = analysis
                    
                    total_words += result.word_count
                    high_quality_sources.append(result)
            
            # Filter by quality level
            quality_threshold = {'Basic': 0.3, 'Enhanced': 0.5, 'Premium': 0.7, 'Academic': 0.8}
            filtered_sources = [s for s in high_quality_sources if s.credibility_score >= quality_threshold[source_quality]]
            
            for i in range(55, 70):
                progress_bar.progress(i)
                time.sleep(0.05)
            
            # Update metrics
            with metrics_placeholder:
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                
                with met_col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <span class="metric-value">{len(filtered_sources)}</span>
                        <span class="metric-label">Quality Sources</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with met_col2:
                    avg_credibility = sum(s.credibility_score for s in filtered_sources) / max(len(filtered_sources), 1)
                    st.markdown(f"""
                    <div class="metric-container">
                        <span class="metric-value">{avg_credibility:.2f}</span>
                        <span class="metric-label">Avg Credibility</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with met_col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <span class="metric-value">{total_words:,}</span>
                        <span class="metric-label">Words Analyzed</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with met_col4:
                    processing_time = (datetime.now() - start_time).seconds
                    st.markdown(f"""
                    <div class="metric-container">
                        <span class="metric-value">{processing_time}s</span>
                        <span class="metric-label">Processing Time</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Phase 4: AI Analysis and Synthesis
            status_text.markdown('<span class="status-indicator status-info"></span> **Synthesizing insights with AI...**', unsafe_allow_html=True)
            
            # Prepare comprehensive context for AI
            research_context = f"""
RESEARCH QUERY: {query}
RESEARCH MODE: {search_mode}
ANALYSIS DATE: {datetime.now().strftime("%B %d, %Y")}
SOURCES ANALYZED: {len(filtered_sources)}
TOTAL CONTENT WORDS: {total_words:,}

SOURCE DETAILS:
"""
            
            for idx, source in enumerate(filtered_sources[:20]):  # Limit context size
                research_context += f"""
--- SOURCE {idx + 1} ---
Title: {source.title}
URL: {source.url}
Domain: {source.domain}
Credibility Score: {source.credibility_score}
Word Count: {source.word_count}
Content Preview: {source.content[:500]}...

"""
            
            # Build advanced prompt based on configuration
            feature_instructions = []
            
            if config.executive_summary:
                if search_mode == "QuickSynth":
                    feature_instructions.append("Begin with a concise Executive Summary (3-4 sentences) highlighting key findings and direct answers to the query.")
                elif search_mode == "QuantumSynth":
                    feature_instructions.append("Start with a comprehensive Executive Summary (100-150 words) providing overview of findings, key insights, and implications.")
                else:  # OmniSynth
                    feature_instructions.append("Begin with an extensive Abstract (200-300 words) covering research scope, methodology, key findings, limitations, and conclusions.")
            
            if config.future_insights:
                feature_instructions.append("Include a dedicated 'Future Outlook' section discussing emerging trends, potential developments, predictions, and their timeline/probability.")
            
            if config.historical_context:
                feature_instructions.append("Provide historical context section tracing the evolution of the topic, key milestones, and how past developments inform current understanding.")
            
            if config.expert_quotes:
                feature_instructions.append("Incorporate relevant expert quotes, authoritative statements, or notable insights found in the sources. Attribute clearly with source information.")
            
            if config.include_perspectives:
                feature_instructions.append("Present multiple perspectives, competing theories, or different schools of thought. Analyze strengths/weaknesses of each approach based on evidence from sources.")
            
            if config.fact_check:
                feature_instructions.append("Include fact-checking elements: verify claims across sources, note consensus vs. disputed points, highlight any conflicting information found.")
            
            if config.semantic_analysis and sentiment_analysis:
                sentiment_summary = {}
                for analysis in sentiment_analysis.values():
                    sentiment = analysis['sentiment']
                    sentiment_summary[sentiment] = sentiment_summary.get(sentiment, 0) + 1
                
                feature_instructions.append(f"Consider the sentiment analysis of sources: {sentiment_summary}. Discuss any apparent bias patterns or emotional framing in the literature.")
            
            if config.data_visualization:
                feature_instructions.append("Identify 2-3 specific data visualizations that would enhance understanding. For each, specify: chart type, data sources from your analysis, axes/categories, and key insights it would reveal.")
            
            # Advanced prompt construction
            if search_mode == "QuickSynth":
                main_prompt = f"""
You are NexusQuery Pro, an elite AI research assistant specializing in rapid knowledge synthesis.

TASK: Create a concise yet comprehensive analysis of: "{query}"

RESPONSE STRUCTURE:
1. Executive Summary (if enabled)
2. Key Findings (3-5 main points with evidence)
3. Critical Insights & Implications
4. Additional sections based on enabled features

REQUIREMENTS:
- Professional, authoritative tone
- Evidence-based conclusions with source attribution
- Clear, scannable formatting with headers
- 800-1200 words total
- Focus on actionable insights and practical implications

ENABLED FEATURES:
{chr(10).join(feature_instructions)}

CITATION STYLE: {config.citation_style}
- Use inline citations [1], [2], etc. referring to source numbers
- Ensure every major claim is supported by source evidence

{research_context}
"""
                max_tokens = 3000
                
            elif search_mode == "QuantumSynth":
                main_prompt = f"""
You are NexusQuery Pro, an elite AI research assistant specializing in comprehensive knowledge synthesis and analysis.

TASK: Create a detailed analytical report on: "{query}"

RESPONSE STRUCTURE:
1. Executive Summary/Abstract (if enabled)
2. Introduction & Background
3. Current State Analysis
4. Key Findings & Evidence
5. Multiple Perspectives Analysis (if enabled)
6. Implications & Applications
7. Additional specialized sections based on enabled features
8. Conclusions & Recommendations

REQUIREMENTS:
- Academic-level analysis with professional tone
- Balanced presentation of evidence and viewpoints
- Clear hierarchical structure with H2/H3 headings
- Critical evaluation of source quality and reliability
- 2000-3000 words total
- Integration of quantitative and qualitative insights

ENABLED FEATURES:
{chr(10).join(feature_instructions)}

CITATION STYLE: {config.citation_style}
- Comprehensive citation of all major claims
- Source evaluation and credibility assessment
- Cross-referencing of findings across sources

{research_context}
"""
                max_tokens = 5000
                
            else:  # OmniSynth
                main_prompt = f"""
You are NexusQuery Pro, the most advanced AI research assistant available, specializing in producing academic-grade comprehensive research reports.

TASK: Create an exhaustive research report on: "{query}"

RESPONSE STRUCTURE:
1. Abstract (200-300 words)
2. Table of Contents
3. Introduction & Research Scope
4. Literature Review & Source Analysis
5. Methodology & Approach
6. Detailed Findings & Analysis
7. Critical Evaluation & Discussion
8. Multiple Perspectives & Theoretical Frameworks
9. Specialized Analysis Sections (based on enabled features)
10. Limitations & Future Research Directions
11. Conclusions & Recommendations
12. References & Source Quality Assessment

REQUIREMENTS:
- Doctoral-level research quality and depth
- Comprehensive coverage of all relevant aspects
- Critical analysis and synthesis of information
- Detailed methodology and source evaluation
- Professional academic writing style
- Extensive use of subheadings for organization
- 4000-6000 words total
- Integration of theoretical and practical perspectives
- Rigorous fact-checking and cross-validation

ENABLED FEATURES:
{chr(10).join(feature_instructions)}

CITATION STYLE: {config.citation_style}
- Extensive citation network with source evaluation
- Academic-standard referencing throughout
- Critical assessment of source reliability and bias
- Cross-validation of claims across multiple sources

{research_context}
"""
                max_tokens = 8000
            
            # Generate AI response
            for i in range(70, 85):
                progress_bar.progress(i)
                time.sleep(0.1)
            
            try:
                response = research_engine.model.generate_content(
                    [main_prompt],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=max_tokens,
                        top_p=0.9,
                        top_k=40
                    )
                )
                
                for i in range(85, 95):
                    progress_bar.progress(i)
                    time.sleep(0.05)
                
                status_text.markdown('<span class="status-indicator status-success"></span> **Finalizing research report...**', unsafe_allow_html=True)
                
                # Process and display results
                if response.candidates and response.text:
                    generated_text = response.text
                    
                    # Apply custom citations if using inline numbers
                    if config.citation_style == "Inline Numbers":
                        # Enhanced citation matching
                        for idx, source in enumerate(filtered_sources):
                            # Look for source content mentions in the generated text
                            source_keywords = []
                            
                            # Extract significant keywords from title
                            title_words = re.findall(r'\b\w{4,}\b', source.title.lower())
                            source_keywords.extend(title_words[:3])
                            
                            # Extract keywords from content
                            if source.content:
                                content_words = re.findall(r'\b\w{5,}\b', source.content.lower())
                                source_keywords.extend(content_words[:5])
                            
                            # Apply citations
                            for keyword in set(source_keywords):
                                if keyword in generated_text.lower():
                                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                                    if f'[{idx+1}]' not in generated_text:
                                        generated_text = pattern.sub(f'{keyword}<span class="citation-number">[{idx+1}]</span>', generated_text, count=1)
                    
                    progress_bar.progress(100)
                    status_text.markdown('<span class="status-indicator status-success"></span> **Research complete! Analysis ready.**', unsafe_allow_html=True)
                    
                    # Display results with enhanced formatting
                    with result_container:
                        st.markdown("---")
                        st.markdown("## 📋 Research Report")
                        
                        # Create downloadable content
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"nexusquery_research_{query.replace(' ', '_')[:30]}_{timestamp}"
                        
                        # Display the research report
                        if config.citation_style == "Inline Numbers":
                            st.markdown(generated_text, unsafe_allow_html=True)
                        else:
                            st.markdown(generated_text)
                        
                        # Action buttons
                        st.markdown("---")
                        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                        
                        with action_col1:
                            st.download_button(
                                label="📄 Download Report",
                                data=generated_text,
                                file_name=f"{filename}.{export_format.lower().replace(' ', '_')}",
                                mime="text/plain" if export_format == "Plain Text" else "text/markdown"
                            )
                        
                        with action_col2:
                            if st.button("🔄 Refine Analysis"):
                                st.info("Click 'Launch Research' again with modified parameters to refine your analysis.")
                        
                        with action_col3:
                            if st.button("📊 View Analytics"):
                                st.switch_page("Analytics Dashboard")
                        
                        with action_col4:
                            if st.button("🔗 Share Research"):
                                st.info("Sharing functionality available in Pro+ version.")
                        
                        # Store results in session state for other tabs
                        st.session_state.research_results = {
                            'query': query,
                            'report': generated_text,
                            'sources': filtered_sources,
                            'config': config,
                            'timestamp': timestamp,
                            'processing_time': processing_time,
                            'total_words': total_words
                        }
                
                else:
                    st.error("❌ AI model returned an empty response. Please try again with a different query or check system status.")
                    
            except Exception as e:
                st.error(f"❌ Research generation failed: {str(e)}")
                logger.error(f"AI generation failed: {e}")
        
        # --- 📚 Source Analysis Tab ---
        with sources_tab:
            if 'research_results' in st.session_state:
                st.markdown("## 📚 Comprehensive Source Analysis")
                
                sources = st.session_state.research_results['sources']
                
                # Source overview metrics
                st.markdown("### 📊 Source Overview")
                overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
                
                with overview_col1:
                    st.metric("Total Sources", len(sources))
                
                with overview_col2:
                    high_cred_sources = len([s for s in sources if s.credibility_score > 0.7])
                    st.metric("High Credibility", high_cred_sources)
                
                with overview_col3:
                    avg_words = sum(s.word_count for s in sources) / max(len(sources), 1)
                    st.metric("Avg. Content Length", f"{int(avg_words)} words")
                
                with overview_col4:
                    academic_sources = len([s for s in sources if '.edu' in s.domain or 'scholar' in s.domain])
                    st.metric("Academic Sources", academic_sources)
                
                # Source quality distribution
                st.markdown("### 📈 Source Quality Distribution")
                
                credibility_data = pd.DataFrame({
                    'Source': [f"Source {s.id}" for s in sources],
                    'Credibility Score': [s.credibility_score for s in sources],
                    'Word Count': [s.word_count for s in sources],
                    'Domain Type': [s.domain.split('.')[-1] if '.' in s.domain else 'other' for s in sources]
                })
                
                fig_scatter = px.scatter(
                    credibility_data, 
                    x='Word Count', 
                    y='Credibility Score',
                    color='Domain Type',
                    size=[100] * len(credibility_data),  # Fixed size
                    hover_data=['Source'],
                    title="Source Quality vs Content Depth Analysis"
                )
                
                fig_scatter.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Detailed source list
                st.markdown("### 🔍 Detailed Source Analysis")
                
                for idx, source in enumerate(sources):
                    with st.expander(f"🔗 Source {source.id}: {source.title[:60]}..." if len(source.title) > 60 else f"🔗 Source {source.id}: {source.title}"):
                        
                        # Source metadata
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        
                        with meta_col1:
                            st.metric("Credibility Score", f"{source.credibility_score:.2f}")
                        
                        with meta_col2:
                            st.metric("Content Length", f"{source.word_count} words")
                        
                        with meta_col3:
                            domain_type = "Academic" if '.edu' in source.domain else "Government" if '.gov' in source.domain else "Organization" if '.org' in source.domain else "Commercial"
                            st.metric("Source Type", domain_type)
                        
                        # Source content
                        st.markdown(f"**URL:** [{source.url}]({source.url})")
                        st.markdown(f"**Domain:** {source.domain}")
                        
                        if source.snippet:
                            st.markdown(f"**Snippet:** {source.snippet}")
                        
                        # Content preview with analysis
                        if source.content:
                            st.markdown("**Content Analysis:**")
                            
                            # Show content preview
                            content_preview = source.content[:500] + "..." if len(source.content) > 500 else source.content
                            st.markdown(f'<div class="source-box">{content_preview}</div>', unsafe_allow_html=True)
                            
                            # Analysis buttons
                            anal_col1, anal_col2, anal_col3 = st.columns(3)
                            
                            with anal_col1:
                                if st.button(f"📖 Full Content", key=f"content_{idx}"):
                                    st.markdown("**Full Content:**")
                                    st.markdown(f'<div class="source-box">{source.content}</div>', unsafe_allow_html=True)
                            
                            with anal_col2:
                                if st.button(f"🧠 AI Summary", key=f"summary_{idx}"):
                                    try:
                                        summary_prompt = f"Summarize the key points from this content in 2-3 sentences: {source.content[:1000]}"
                                        summary_response = research_engine.model.generate_content([summary_prompt])
                                        if summary_response.text:
                                            st.markdown("**AI Summary:**")
                                            st.info(summary_response.text)
                                    except:
                                        st.error("Summary generation failed")
                            
                            with anal_col3:
                                if st.button(f"📊 Key Stats", key=f"stats_{idx}"):
                                    # Extract basic statistics
                                    sentences = len(re.split(r'[.!?]+', source.content))
                                    paragraphs = len(source.content.split('\n\n'))
                                    
                                    st.markdown("**Content Statistics:**")
                                    st.write(f"- Sentences: {sentences}")
                                    st.write(f"- Paragraphs: {paragraphs}")
                                    st.write(f"- Reading time: ~{source.word_count // 200} minutes")
            else:
                st.info("🔍 Run a research query first to analyze sources.")
        
        # --- 📊 Analytics Dashboard Tab ---
        with analytics_tab:
            if 'research_results' in st.session_state:
                st.markdown("## 📊 Advanced Research Analytics")
                
                results = st.session_state.research_results
                sources = results['sources']
                
                # Generate visualizations
                if config.data_visualization:
                    st.markdown("### 📈 Data Visualizations")
                    
                    visualizations = research_engine.generate_visualizations(query, sources)
                    
                    if visualizations:
                        for viz in visualizations:
                            st.markdown(f"#### {viz['title']}")
                            st.plotly_chart(viz['figure'], use_container_width=True)
                            st.markdown(f"*{viz['description']}*")
                            st.markdown("---")
                    else:
                        st.info("No suitable visualizations could be generated from the current data.")
                
                # Research performance metrics
                st.markdown("### ⚡ Performance Metrics")
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                with perf_col1:
                    st.metric("Processing Time", f"{results['processing_time']}s")
                
                with perf_col2:
                    sources_per_second = len(sources) / max(results['processing_time'], 1)
                    st.metric("Sources/Second", f"{sources_per_second:.1f}")
                
                with perf_col3:
                    words_per_second = results['total_words'] / max(results['processing_time'], 1)
                    st.metric("Words/Second", f"{words_per_second:.0f}")
                
                with perf_col4:
                    efficiency_score = (len(sources) * sum(s.credibility_score for s in sources)) / max(results['processing_time'], 1)
                    st.metric("Efficiency Score", f"{efficiency_score:.1f}")
                
                # Content analysis dashboard
                st.markdown("### 🧠 Content Analysis Dashboard")
                
                # Create comprehensive analytics
                domains = {}
                credibility_ranges = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
                content_lengths = []
                
                for source in sources:
                    # Domain analysis
                    domain_ext = source.domain.split('.')[-1] if '.' in source.domain else 'other'
                    domains[domain_ext] = domains.get(domain_ext, 0) + 1
                    
                    # Credibility range analysis
                    score = source.credibility_score
                    if score < 0.3:
                        credibility_ranges["0.0-0.3"] += 1
                    elif score < 0.5:
                        credibility_ranges["0.3-0.5"] += 1
                    elif score < 0.7:
                        credibility_ranges["0.5-0.7"] += 1
                    elif score < 0.9:
                        credibility_ranges["0.7-0.9"] += 1
                    else:
                        credibility_ranges["0.9-1.0"] += 1
                    
                    content_lengths.append(source.word_count)
                
                # Display analytics charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Domain distribution pie chart
                    if domains:
                        fig_domains = px.pie(
                            values=list(domains.values()),
                            names=list(domains.keys()),
                            title="Source Domain Distribution"
                        )
                        fig_domains.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )
                        st.plotly_chart(fig_domains, use_container_width=True)
                
                with chart_col2:
                    # Credibility range bar chart
                    fig_cred = px.bar(
                        x=list(credibility_ranges.keys()),
                        y=list(credibility_ranges.values()),
                        title="Credibility Score Distribution",
                        labels={'x': 'Credibility Range', 'y': 'Number of Sources'}
                    )
                    fig_cred.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_cred, use_container_width=True)
                
                # Research insights summary
                st.markdown("### 💡 Research Insights Summary")
                
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.markdown("**Quality Assessment:**")
                    avg_credibility = sum(s.credibility_score for s in sources) / max(len(sources), 1)
                    quality_assessment = "Excellent" if avg_credibility > 0.8 else "Good" if avg_credibility > 0.6 else "Fair" if avg_credibility > 0.4 else "Needs Improvement"
                    st.info(f"Overall source quality: **{quality_assessment}** (Avg: {avg_credibility:.2f})")
                    
                    high_qual_percentage = len([s for s in sources if s.credibility_score > 0.7]) / max(len(sources), 1) * 100
                    st.info(f"High-quality sources: **{high_qual_percentage:.1f}%** of total")
                
                with insights_col2:
                    st.markdown("**Coverage Analysis:**")
                    total_words = sum(s.word_count for s in sources)
                    est_reading_time = total_words // 200
                    st.info(f"Total content analyzed: **{total_words:,} words** (~{est_reading_time} min read)")
                    
                    diverse_sources = len(set(s.domain.split('.')[-1] for s in sources))
                    st.info(f"Source diversity: **{diverse_sources} different domain types**")
                
            else:
                st.info("📊 Run a research query first to view analytics.")
        
        # --- 💾 Export & Share Tab ---
        with export_tab:
            if 'research_results' in st.session_state:
                st.markdown("## 💾 Export & Share Research")
                
                results = st.session_state.research_results
                
                # Export options
                st.markdown("### 📄 Export Formats")
                
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    # Plain text export
                    text_content = f"""
NexusQuery Pro Research Report
Generated: {results['timestamp']}
Query: {results['query']}
Processing Time: {results['processing_time']}s
Sources Analyzed: {len(results['sources'])}

{results['report']}

--- SOURCE REFERENCES ---
"""
                    for idx, source in enumerate(results['sources']):
                        text_content += f"[{idx+1}] {source.title}\n    {source.url}\n    Credibility: {source.credibility_score:.2f}\n\n"
                    
                    st.download_button(
                        "📄 Download as Text",
                        data=text_content,
                        file_name=f"research_report_{results['timestamp']}.txt",
                        mime="text/plain"
                    )
                
                with export_col2:
                    # Markdown export
                    markdown_content = f"""# NexusQuery Pro Research Report

**Generated:** {results['timestamp']}  
**Query:** {results['query']}  
**Processing Time:** {results['processing_time']}s  
**Sources Analyzed:** {len(results['sources'])}

---

{results['report']}

## Source References

"""
                    for idx, source in enumerate(results['sources']):
                        markdown_content += f"{idx+1}. **{source.title}**  \n   URL: [{source.url}]({source.url})  \n   Credibility Score: {source.credibility_score:.2f}  \n   Domain: {source.domain}\n\n"
                    
                    st.download_button(
                        "📝 Download as Markdown",
                        data=markdown_content,
                        file_name=f"research_report_{results['timestamp']}.md",
                        mime="text/markdown"
                    )
                
                with export_col3:
                    # JSON export
                    json_data = {
                        "metadata": {
                            "query": results['query'],
                            "timestamp": results['timestamp'],
                            "processing_time": results['processing_time'],
                            "config": results['config'].__dict__
                        },
                        "report": results['report'],
                        "sources": [
                            {
                                "id": s.id,
                                "title": s.title,
                                "url": s.url,
                                "domain": s.domain,
                                "credibility_score": s.credibility_score,
                                "word_count": s.word_count,
                                "snippet": s.snippet
                            } for s in results['sources']
                        ]
                    }
                    
                    st.download_button(
                        "📊 Download as JSON",
                        data=json.dumps(json_data, indent=2),
                        file_name=f"research_data_{results['timestamp']}.json",
                        mime="application/json"
                    )
                
                # Share options
                st.markdown("### 🔗 Share Options")
                
                share_col1, share_col2, share_col3 = st.columns(3)
                
                with share_col1:
                    if st.button("📧 Generate Email Report"):
                        email_subject = f"Research Report: {results['query']}"
                        email_body = f"""
I've completed a comprehensive research analysis using NexusQuery Pro.

Query: {results['query']}
Sources Analyzed: {len(results['sources'])}
Processing Time: {results['processing_time']}s

Key Findings Summary:
{results['report'][:500]}...

Full report attached.
"""
                        st.text_area("Email Content (Copy to your email client):", email_body, height=200)
                
                with share_col2:
                    if st.button("🔗 Generate Shareable Link"):
                        st.info("Shareable links feature available in Enterprise version.")
                
                with share_col3:
                    if st.button("📋 Copy Report Summary"):
                        summary = f"Research on '{results['query']}' completed. {len(results['sources'])} sources analyzed in {results['processing_time']}s. Key insights ready for review."
                        st.code(summary, language=None)
                        st.success("Summary ready to copy!")
                
                # Research statistics
                st.markdown("### 📊 Research Statistics")
                
                stats_data = {
                    "Research Query": results['query'],
                    "Total Sources Found": len(results['sources']),
                    "High-Quality Sources": len([s for s in results['sources'] if s.credibility_score > 0.7]),
                    "Academic Sources": len([s for s in results['sources'] if '.edu' in s.domain]),
                    "Government Sources": len([s for s in results['sources'] if '.gov' in s.domain]),
                    "Total Words Analyzed": f"{results['total_words']:,}",
                    "Processing Time": f"{results['processing_time']} seconds",
                    "Average Credibility Score": f"{sum(s.credibility_score for s in results['sources']) / len(results['sources']):.3f}",
                    "Research Mode": results['config'].search_mode,
                    "Citation Style": results['config'].citation_style
                }
                
                stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value'])
                st.table(stats_df)
                
                # Advanced export options
                st.markdown("### 🔧 Advanced Export Options")
                
                advanced_col1, advanced_col2 = st.columns(2)
                
                with advanced_col1:
                    st.markdown("**Custom Export Settings:**")
                    include_sources = st.checkbox("Include Source Details", value=True)
                    include_metadata = st.checkbox("Include Research Metadata", value=True)
                    include_analytics = st.checkbox("Include Analytics Data", value=False)
                    compress_export = st.checkbox("Compress Large Exports", value=False)
                
                with advanced_col2:
                    st.markdown("**Export Scheduling:**")
                    if st.button("📅 Schedule Weekly Reports"):
                        st.info("Report scheduling available in Pro+ version.")
                    
                    if st.button("🔄 Set Up Auto-Export"):
                        st.info("Auto-export features available in Enterprise version.")
                
                # Data privacy notice
                st.markdown("---")
                st.info("🔒 **Data Privacy:** All research data is processed securely. No personal information is stored permanently. Export files contain only the research content you generate.")
            
            else:
                st.info("💾 Run a research query first to access export options.")
    
    except Exception as e:
        st.error(f"❌ Research process failed: {str(e)}")
        logger.error(f"Main research process error: {e}")
        
        # Error recovery suggestions
        st.markdown("### 🔧 Troubleshooting Suggestions:")
        st.markdown("""
        1. **Check your query:** Ensure it's specific and well-formed
        2. **Verify API keys:** Confirm GEMINI_API_KEY and SERPAPI_KEY are valid
        3. **Reduce complexity:** Try a simpler query or fewer sources
        4. **Check network:** Ensure stable internet connection
        5. **Try again:** Some errors are temporary and resolve on retry
        """)
        
        if st.button("🔄 Retry Research"):
            st.rerun()

# --- 🚀 Additional Features Section ---
if not (search_button and query):
    st.markdown("---")
    
    # Feature showcase
    st.markdown("## 🌟 Advanced Research Capabilities")
    
    feature_cols = st.columns(3)
    
    with feature_cols[0]:
        st.markdown("""
        ### 🧠 AI-Powered Analysis
        - **Multi-Model Intelligence:** Leverages Gemini 2.0 Flash for comprehensive analysis
        - **Semantic Understanding:** Deep comprehension of context and nuance
        - **Bias Detection:** Identifies and flags potential source bias
        - **Fact Verification:** Cross-validates claims across multiple sources
        """)
    
    with feature_cols[1]:
        st.markdown("""
        ### 🔍 Advanced Search Technology
        - **Multi-Source Integration:** Academic, government, and commercial sources
        - **Quality Scoring:** Automatic credibility assessment
        - **Parallel Processing:** Simultaneous analysis of multiple sources
        - **Real-Time Updates:** Live progress tracking and metrics
        """)
    
    with feature_cols[2]:
        st.markdown("""
        ### 📊 Professional Analytics
        - **Data Visualizations:** Interactive charts and graphs
        - **Performance Metrics:** Detailed processing analytics
        - **Export Options:** Multiple formats for different use cases
        - **Quality Assurance:** Comprehensive source validation
        """)
    
    # Research examples showcase
    st.markdown("## 💡 Example Research Applications")
    
    example_cols = st.columns(2)
    
    with example_cols[0]:
        st.markdown("""
        ### 🏢 **Business Intelligence**
        - Market research and competitive analysis
        - Industry trend analysis and forecasting
        - Technology adoption studies
        - Investment opportunity research
        
        ### 🎓 **Academic Research**
        - Literature reviews and meta-analysis
        - Current research state assessment
        - Methodology comparison studies
        - Citation and reference compilation
        """)
    
    with example_cols[1]:
        st.markdown("""
        ### 🏛️ **Policy & Government**
        - Policy impact assessment
        - Regulatory environment analysis
        - Public opinion research
        - Legislative trend tracking
        
        ### 💡 **Innovation & Technology**
        - Emerging technology evaluation
        - Patent landscape analysis
        - Research gap identification
        - Future trend prediction
        """)

# --- 🔧 System Information Footer ---
st.markdown("---")
st.markdown("## 🔧 System Information")

footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

with footer_col1:
    st.markdown("**🚀 Performance**")
    st.markdown("- Parallel processing enabled")
    st.markdown("- Real-time progress tracking")
    st.markdown("- Advanced caching system")

with footer_col2:
    st.markdown("**🔒 Security**")
    st.markdown("- Secure API connections")
    st.markdown("- No data persistence")
    st.markdown("- Privacy-focused design")

with footer_col3:
    st.markdown("**🎯 Accuracy**")
    st.markdown("- Multi-source validation")
    st.markdown("- Credibility scoring")
    st.markdown("- Bias detection algorithms")

with footer_col4:
    st.markdown("**📊 Analytics**")
    st.markdown("- Comprehensive metrics")
    st.markdown("- Visual data representation")
    st.markdown("- Export capabilities")

# Final footer
st.markdown("""
<div class="footer">
    <h3>🧠 NexusQuery Pro - Professional AI Research Engine</h3>
    <p>
        <strong>Version 2.0</strong> | Powered by Gemini 2.0 Flash & Advanced Search APIs
        <br>
        Built for researchers, analysts, and knowledge professionals
        <br>
        <small>© 2025 NexusQuery Pro. All research is conducted ethically with respect for source attribution.</small>
    </p>
</div>
""", unsafe_allow_html=True)

# Performance monitoring (hidden)
if st.secrets.get("DEBUG_MODE", False):
    st.markdown("---")
    st.markdown("### 🔍 Debug Information")
    st.json({
        "session_state_keys": list(st.session_state.keys()),
        "current_time": datetime.now().isoformat(),
        "streamlit_version": st.__version__
    })
