import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from nltk import FreqDist

st.set_page_config(page_title="GATEWAYS-2025 Dashboard", layout="wide")
st.title("GATEWAYS-2025 National Fest Analysis")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    df.columns = df.columns.str.strip()

    st.header("Participation Trends")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Event-wise Participation")
        event_counts = df['Event Name'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.bar(event_counts.index, event_counts.values)
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    with col2:
        st.subheader("College-wise Participation")
        college_counts = df['College'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.bar(college_counts.index, college_counts.values)
        plt.xticks(rotation=90)
        st.pyplot(fig2)

    import geopandas as gpd

    st.header("State-wise Participation (India Map)")

    state_counts = df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']

    india_map = gpd.read_file("india_states.shp")

    india_map['State'] = india_map['State'].str.strip()
    state_counts['State'] = state_counts['State'].str.strip()

    merged = india_map.merge(state_counts, on='State', how='left')
    merged['Count'] = merged['Count'].fillna(0)

    fig, ax = plt.subplots(figsize=(10,10))
    merged.plot(column='Count', cmap='OrRd', legend=True, ax=ax)

    plt.title("State-wise Participation in GATEWAYS-2025")
    plt.axis('off')

    st.pyplot(fig)

    st.header("Feedback & Rating Analysis")

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    df['Cleaned Feedback'] = df['Feedback on Fest'].apply(clean_text)

    words = " ".join(df['Cleaned Feedback']).split()
    word_freq = FreqDist(words).most_common(10)

    st.subheader("Top Words in Feedback")
    st.write(word_freq)

    st.subheader("Ratings Distribution")
    rating_counts = df['Rating'].value_counts().sort_index()
    fig3, ax3 = plt.subplots()
    ax3.bar(rating_counts.index, rating_counts.values)
    st.pyplot(fig3)

    st.subheader("Average Rating per Event")
    avg_rating = df.groupby('Event Name')['Rating'].mean()
    fig4, ax4 = plt.subplots()
    ax4.bar(avg_rating.index, avg_rating.values)
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    st.header("Interactive Dashboard")

    selected_event = st.selectbox("Select Event", df['Event Name'].unique())
    filtered_df = df[df['Event Name'] == selected_event]

    st.write(f"Participants in {selected_event}:")
    st.dataframe(filtered_df)

    st.metric("Average Rating", round(filtered_df['Rating'].mean(), 2))
    st.metric("Total Participants", len(filtered_df))

else:
    st.warning("Please upload a CSV file to proceed.")