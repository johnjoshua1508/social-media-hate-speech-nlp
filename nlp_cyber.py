import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from collections import defaultdict
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Hate Speech Analysis", layout="wide")

# Load the summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

# Function to generate word cloud
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    plt.tight_layout(pad=0)
    return fig

# Function to analyze victims
def analyze_victims(df):
    demographic_columns = {
        'Race': 'Race',
        'Religion': 'Religion',
        'Gender': 'Gender',
        'Sexual Orientation': 'Sexual Orientation',
        'Miscellaneous': 'Miscellaneous'
    }
    
    demographics = {}
    for display_name, col_name in demographic_columns.items():
        demographics[display_name] = df[col_name].value_counts()
    
    hate_speech_df = df[df['label'] == 'hatespeech']
    
    target_summaries = defaultdict(list)
    wordclouds = {}
    
    for category_display, category_col in demographic_columns.items():
        exclude_values = ['No_race', 'No_gender', 'No_orientation', 'None', 'Nonreligious']
        all_comments = []
        
        for value in demographics[category_display].index:
            if value not in exclude_values:
                relevant_comments = hate_speech_df[hate_speech_df[category_col] == value]['comment'].tolist()
                if relevant_comments:
                    all_comments.extend(relevant_comments)
                    sample = ' '.join(relevant_comments[:5])
                    if len(sample) > 1000:
                        sample = sample[:1000]
                    
                    try:
                        summary = summarizer(sample, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                    except Exception as e:
                        summary = "Summary generation failed"
                    
                    target_summaries[category_display].append({
                        'group': value,
                        'count': len(relevant_comments),
                        'summary': summary
                    })
        
        # Generate word cloud for each category
        if all_comments:
            preprocessed_text = ' '.join(map(preprocess_text, all_comments))
            wordclouds[category_display] = generate_wordcloud(preprocessed_text, f"{category_display} Hate Speech")

    return target_summaries, wordclouds

# Function to display results
def display_results(summaries, wordclouds, category):
    st.header(f"{category} Based Targeting")
    
    # Display results in a table
    results_df = pd.DataFrame(summaries[category])
    results_df = results_df.sort_values('count', ascending=False)
    st.table(results_df[['group', 'count']])

    # Display summaries
    for result in summaries[category]:
        with st.expander(f"Summary for {result['group']}"):
            st.write(result['summary'])

    # Display word cloud
    if category in wordclouds:
        st.pyplot(wordclouds[category])

# Main Streamlit app
def main():
    st.title("NLP System for Summarization of Cyber Crime Victims on Social Media")

    # Sidebar
    st.sidebar.title("Navigation")
    
    # File uploader in sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Process data if not already processed
        if 'processed_data' not in st.session_state:
            df = pd.read_csv(uploaded_file)
            summaries, wordclouds = analyze_victims(df)
            st.session_state['processed_data'] = {
                'df': df,
                'summaries': summaries,
                'wordclouds': wordclouds
            }
            st.sidebar.success("Data processed successfully!")

        # Navigation options
        pages = ["Overall Statistics", "Race", "Religion", "Gender", "Sexual Orientation", "Miscellaneous", "Data Preview"]
        choice = st.sidebar.radio("Go to", pages)

        # Display based on navigation choice
        if choice == "Overall Statistics":
            st.header("Overall Statistics")
            df = st.session_state['processed_data']['df']
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Posts", len(df))
            col2.metric("Hate Speech Posts", len(df[df['label'] == 'hatespeech']))
            col3.metric("Offensive Posts", len(df[df['label'] == 'offensive']))
        elif choice in ["Race", "Religion", "Gender", "Sexual Orientation", "Miscellaneous"]:
            display_results(st.session_state['processed_data']['summaries'], 
                            st.session_state['processed_data']['wordclouds'], 
                            choice)
        elif choice == "Data Preview":
            st.header("Data Preview")
            st.dataframe(st.session_state['processed_data']['df'].head())

    else:
        st.write("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()