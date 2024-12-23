import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from langchain.schema import Document
import traceback

# Streamlit App Setup
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Sidebar for inputs
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password", help="Enter your Groq API Key.")

# Main input
generic_url = st.text_input("Enter URL (YouTube or Website)", value="", help="Paste a YouTube or website URL here.")

# Prompt Template
prompt_template = """
Provide a concise summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarization Button
if st.button("Summarize the Content"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the API key and URL to proceed.")
    elif not validators.url(generic_url):
        st.error("Invalid URL. Please enter a valid YouTube or website URL.")
    else:
        try:
            with st.spinner("Processing..."):
                # Load content from the URL
                docs = None
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        # Try loading YouTube content
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                        docs = loader.load()
                    except Exception as yt_error:
                        # Fallback: Use YouTube Transcript API
                        st.warning("YouTubeLoader failed, attempting to use YouTube Transcript API...")
                        try:
                            video_id = YouTube(generic_url).video_id
                            transcript = YouTubeTranscriptApi.get_transcript(video_id)
                            docs = [Document(page_content=" ".join([entry["text"] for entry in transcript]))]
                        except Exception as transcript_error:
                            st.error(f"Failed to retrieve transcript: {str(transcript_error)}")
                            docs = None
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                if not docs:
                    st.error("No content could be extracted from the provided URL.")
                else:
                    # Initialize ChatGroq model
                    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

                    # Summarization Chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.success("Summary generated successfully!")
                    st.write(output_summary)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(traceback.format_exc())
