import streamlit as st

# Set page config - MUST be the first command
st.set_page_config(page_title="Qualitative Analysis Chatbot", layout="wide")

import pandas as pd
import docx
import openai
import google.generativeai as genai
from textblob import TextBlob
import nltk
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import InferenceClient

# Initialize NLTK
@st.cache_resource
def initialize_nltk():
    nltk.download('punkt')
    return True

initialize_nltk()

# OpenAI API Function (Fixed)
def process_with_openai(prompt, text, task):
    try:
        client = openai.Client()  # ✅ Corrected client initialization
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a qualitative analysis expert."},
                {"role": "user", "content": f"Task: {task}\n\nQuestion: {prompt}\n\nText to analyze: {text}"}
            ],
            api_key=st.session_state.api_key  # ✅ API key passed correctly
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI API: {str(e)}"

# Google Gemini API Function
def process_with_gemini(prompt, text, task):
    try:
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"Task: {task}\n\nQuestion: {prompt}\n\nText to analyze: {text}")
        return response.text
    except Exception as e:
        return f"Error with Google Gemini API: {str(e)}"

# Deepseek-R1 API Function (Fixed)
def process_with_deepseek(prompt, text, task):
    try:
        client = InferenceClient(token=st.session_state.api_key)  # ✅ API key passed correctly
        response = client.text_generation(
            model="deepseek-ai/DeepSeek-R1",
            prompt=f"Task: {task}\n\nQuestion: {prompt}\n\nText to analyze: {text}",
            max_new_tokens=500
        )
        return response
    except Exception as e:
        return f"Error with Deepseek API: {str(e)}"

# Main UI Setup
st.title("Qualitative Analysis Chatbot")

# Sidebar Configuration
with st.sidebar:
    st.title("Configuration")
    api_choice = st.radio("Select AI Model:", ["OpenAI", "Google Gemini", "Deepseek-R1"])
    api_key = st.text_input("Enter API Key:", type="password")
    if st.button("Save Configuration"):
        st.session_state.api_choice = api_choice
        st.session_state.api_key = api_key
        st.success(f"Configuration saved for {api_choice}")

# Input Handling
input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
user_question = st.text_area("Enter your question:", height=100)

qualitative_data = ""

if input_method == "Text Input":
    qualitative_data = st.text_area("Paste your qualitative data here:", height=200)
else:
    uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx', 'docx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                qualitative_data = df.to_string()
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                qualitative_data = "\n".join([paragraph.text for paragraph in doc.paragraphs])

            if qualitative_data.strip():
                st.success("File content loaded successfully!")
            else:
                st.warning("The uploaded file appears to be empty. Please upload a valid file.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Perform Analysis
def create_sentiment_plots(sentences):
    sentiment_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
    sentiment_labels = [
        "Very Negative" if s <= -0.6 else "Negative" if -0.6 < s <= -0.2 else "Neutral" if -0.2 < s < 0.2 else "Positive" if 0.2 <= s < 0.6 else "Very Positive"
        for s in sentiment_scores
    ]
    sentiment_distribution = {label: sentiment_labels.count(label) for label in set(sentiment_labels)}
    
    # Bar Chart
    bar_fig = go.Figure([go.Bar(x=list(sentiment_distribution.keys()), y=list(sentiment_distribution.values()))])
    bar_fig.update_layout(title='Sentiment Distribution', xaxis_title='Sentiment Category', yaxis_title='Count')
    
    # Sentiment Flow Chart
    flow_fig = go.Figure([go.Scatter(y=sentiment_scores, mode='lines+markers')])
    flow_fig.update_layout(title='Sentiment Flow', xaxis_title='Sentence Number', yaxis_title='Sentiment Score', yaxis=dict(range=[-1, 1]))
    
    return bar_fig, flow_fig

if st.button("Analyze"):
    if not st.session_state.api_key:
        st.error("Please configure your API key in the sidebar first.")
    elif not user_question.strip():
        st.error("Please enter a valid question.")
    elif not qualitative_data.strip():
        st.error("Please provide a valid text or upload a non-empty file.")
    else:
        with st.spinner("Analyzing your data..."):
            # Sentiment Analysis
            blob = TextBlob(qualitative_data)
            overall_sentiment = blob.sentiment
            sentences = qualitative_data.split('.')
            
            st.metric("Overall Sentiment Score", f"{overall_sentiment.polarity:.2f}")
            st.metric("Overall Subjectivity", f"{overall_sentiment.subjectivity:.2f}")
            
            bar_fig, flow_fig = create_sentiment_plots(sentences)
            st.plotly_chart(bar_fig)
            st.plotly_chart(flow_fig)
            
            # AI Analysis
            st.write("### AI Analysis")
            if st.session_state.api_choice == "OpenAI":
                summary = process_with_openai(user_question, qualitative_data, "Summarize the key themes and main points from the text.")
            elif st.session_state.api_choice == "Google Gemini":
                summary = process_with_gemini(user_question, qualitative_data, "Summarize the key themes and main points from the text.")
            else:
                with st.spinner("Deepseek-R1 is processing your request..."):
                    summary = process_with_deepseek(user_question, qualitative_data, "Summarize the key themes and main points from the text.")
            
            st.write("**Summary:**")
            st.write(summary)
