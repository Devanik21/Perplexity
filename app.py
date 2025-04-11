import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

# --- 🔐 Gemini API Setup ---
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Perplexity-Style Deep Research", layout="wide", page_icon="🧠")
st.title("🧠 Perplexity-Style Deep Research Agent")
st.markdown("Ask a question and get a full-length, research-level answer with real-world insights and future perspectives.")

# --- Helper: Fetch Google Results using SerpAPI ---
def get_search_results(query):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "engine": "google",
        "api_key": st.secrets["SERPAPI_KEY"],
        "num": 20
    }
    response = requests.get(url, params=params)
    data = response.json()
    results = ""
    for result in data.get("organic_results", []):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        results += f"🔸 **{title}**\n{snippet}\n🔗 {link}\n\n"
    return results

# --- Perplexity Deep Research ---
query = st.text_input("🔍 Enter your research query")
if st.button("🔮 Generate Deep Research") and query:
    try:
        with st.spinner("🔍 Searching and compiling research..."):
            search_context = get_search_results(query)

            prompt = f'''
You are a highly intelligent and articulate research assistant.

Using the search snippets below, write an in-depth research article on: "{query}"

The article must be at least **5000 words** and include extensive detail, academic tone, and structured formatting with headings, bullet points, and examples.

Include:
1. A thorough and nuanced explanation of the topic
2. Real-world applications and case studies
3. Current challenges, limitations, and criticisms
4. Ongoing research, future directions, and potential innovations
5. A strong conclusion that synthesizes the core ideas and offers forward-thinking perspectives

Please include citations if possible, and use a formal but clear tone throughout.

Search Snippets:
{search_context}
'''

            response = model.generate_content(
                [prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=8192
                )
            )

            st.markdown("### 📋 AI-Generated Deep Research Report")
            st.write(response.text)

            st.download_button(
                label="📄 Download Full Report",
                data=response.text,
                file_name=f"{query.replace(' ', '_')}_deep_research.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
