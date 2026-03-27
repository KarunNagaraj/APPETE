import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "C5-FestDataset - fest_dataset.csv"
SHAPEFILE_PATH = (
    BASE_DIR
    / "India-State-and-Country-Shapefile-Updated-Jan-2020-master"
    / "India_State_Boundary.shp"
)

st.set_page_config(page_title="GATEWAYS-2025 Dashboard", layout="wide")
plt.style.use("seaborn-v0_8-whitegrid")

st.title("GATEWAYS-2025 National Fest Analysis")
st.caption("A simple overview of participation, feedback, ratings, and event performance.")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

def style_axis(ax):
    ax.set_facecolor("#fbfdff")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

total_participants = len(df)
total_colleges = df["College"].nunique()
total_states = df["State"].nunique()
total_revenue = df["Amount Paid"].sum()
top_event = df["Event Name"].value_counts().idxmax()
top_college = df["College"].value_counts().idxmax()
highest_rated_event = df.groupby("Event Name")["Rating"].mean().idxmax()

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Participants", total_participants)
metric2.metric("Colleges", total_colleges)
metric3.metric("States", total_states)
metric4.metric("Revenue", f"Rs. {total_revenue:,.0f}")

insight1, insight2, insight3 = st.columns(3)
insight1.metric("Top Event", top_event)
insight2.metric("Top College", top_college)
insight3.metric("Highest Rated Event", highest_rated_event)

with st.expander("Dataset Preview"):
    st.dataframe(df, use_container_width=True, height=260)

st.header("Participation Trends")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Event-wise Participation")
    event_counts = df['Event Name'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5.2, 3.4))
    ax1.bar(event_counts.index, event_counts.values, color="#1f77b4", width=0.65)
    ax1.set_ylabel("Participants")
    ax1.tick_params(axis="x", rotation=35)
    style_axis(ax1)
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)

with col2:
    st.subheader("College-wise Participation")
    college_counts = df['College'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.4))
    ax2.bar(college_counts.index, college_counts.values, color="#ff8c42", width=0.65)
    ax2.set_ylabel("Participants")
    ax2.tick_params(axis="x", rotation=90)
    style_axis(ax2)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("State-wise Participation")
    state_counts = df['State'].value_counts()
    fig3, ax3 = plt.subplots(figsize=(5.2, 3.4))
    ax3.bar(state_counts.index, state_counts.values, color="#17a398", width=0.65)
    ax3.set_ylabel("Participants")
    ax3.tick_params(axis="x", rotation=90)
    style_axis(ax3)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

with col4:
    st.subheader("Event Type-wise Participation")
    event_type_counts = df['Event Type'].value_counts()
    fig_type, ax_type = plt.subplots(figsize=(5.2, 3.4))
    ax_type.bar(event_type_counts.index, event_type_counts.values, color="#6c5ce7", width=0.55)
    ax_type.set_ylabel("Participants")
    ax_type.tick_params(axis="x", rotation=0)
    style_axis(ax_type)
    fig_type.tight_layout()
    st.pyplot(fig_type, use_container_width=True)

with st.container():
    st.header("State-wise Participation (India Map)")

    state_counts = df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    state_name_map = {
        'Tamil Nadu': 'Tamilnadu',
        'Telangana': 'Telengana',
    }
    state_counts['State'] = state_counts['State'].str.strip().replace(state_name_map)

    india_map = gpd.read_file(SHAPEFILE_PATH)
    india_map['State_Name'] = india_map['State_Name'].str.strip()

    merged = india_map.merge(state_counts, left_on='State_Name', right_on='State', how='left')
    merged['Count'] = merged['Count'].fillna(0)

    fig, ax = plt.subplots(figsize=(8, 8))
    merged.plot(column='Count', cmap='YlGnBu', legend=True, ax=ax, edgecolor="white", linewidth=0.6)

    plt.title("State-wise Participation in GATEWAYS-2025")
    plt.axis('off')
    st.pyplot(fig, use_container_width=True)

st.header("Feedback & Rating Analysis")

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def remove_stopwords(text):
    return " ".join(word for word in text.split() if word not in stop_words)

def analyze_sentiment(text):
    score = sia.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"

df['Cleaned Feedback'] = df['Feedback on Fest'].apply(clean_text)
df['Processed Feedback'] = df['Cleaned Feedback'].apply(remove_stopwords)
df['Sentiment Score'] = df['Feedback on Fest'].apply(lambda text: sia.polarity_scores(str(text))["compound"])
df['Sentiment'] = df['Sentiment Score'].apply(
    lambda score: "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
)

words = " ".join(df['Processed Feedback']).split()
word_freq = FreqDist(words).most_common(10)
average_sentiment_score = df['Sentiment Score'].mean()

sent1, sent2 = st.columns(2)
sent1.metric("Average Sentiment Score", f"{average_sentiment_score:.2f}")
sent2.metric("Positive Feedback", int((df['Sentiment'] == "Positive").sum()))

st.subheader("Top Words in Feedback")
top_words_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
st.dataframe(top_words_df, use_container_width=True, hide_index=True)

st.subheader("Feedback Sentiment Analysis")
sentiment_counts = df['Sentiment'].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
fig_sentiment, ax_sentiment = plt.subplots(figsize=(6.4, 3.4))
ax_sentiment.bar(sentiment_counts.index, sentiment_counts.values, color=["#2a9d8f", "#8d99ae", "#e76f51"], width=0.6)
ax_sentiment.set_xlabel("Sentiment")
ax_sentiment.set_ylabel("Feedback Count")
style_axis(ax_sentiment)
fig_sentiment.tight_layout()
st.pyplot(fig_sentiment, use_container_width=True)

tab1, tab2 = st.tabs(["Ratings Distribution", "Average Rating per Event"])
with tab1:
    st.subheader("Ratings Distribution")
    rating_counts = df['Rating'].value_counts().sort_index()
    fig3, ax3 = plt.subplots(figsize=(6, 3.4))
    ax3.bar(rating_counts.index, rating_counts.values, color="#577590", width=0.6)
    ax3.set_xlabel("Rating")
    ax3.set_ylabel("Count")
    style_axis(ax3)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

with tab2:
    st.subheader("Average Rating per Event")
    avg_rating = df.groupby('Event Name')['Rating'].mean()
    fig4, ax4 = plt.subplots(figsize=(6, 3.4))
    ax4.bar(avg_rating.index, avg_rating.values, color="#c77dff", width=0.6)
    ax4.set_xlabel("Event Name")
    ax4.set_ylabel("Average Rating")
    ax4.tick_params(axis="x", rotation=35)
    style_axis(ax4)
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)

st.subheader("Word Cloud of Feedback")
feedback_text = " ".join(df['Processed Feedback'])

wordcloud = WordCloud(width=1000, height=420, background_color='white', colormap="viridis").generate(feedback_text)
fig5, ax5 = plt.subplots(figsize=(10, 4.6))
ax5.imshow(wordcloud, interpolation='bilinear')
ax5.axis('off')
st.pyplot(fig5, use_container_width=True)

st.header("Interactive Dashboard")

filter_col1, filter_col2, filter_col3 = st.columns(3)

selected_state = filter_col1.selectbox("Select State", ["All"] + sorted(df['State'].unique()))
state_filtered_df = df if selected_state == "All" else df[df['State'] == selected_state]

selected_college = filter_col2.selectbox("Select College", ["All"] + sorted(state_filtered_df['College'].unique()))
college_filtered_df = state_filtered_df if selected_college == "All" else state_filtered_df[state_filtered_df['College'] == selected_college]

selected_event = filter_col3.selectbox("Select Event", ["All"] + sorted(college_filtered_df['Event Name'].unique()))
filtered_df = college_filtered_df if selected_event == "All" else college_filtered_df[college_filtered_df['Event Name'] == selected_event]

st.subheader("Filtered Participants")
st.dataframe(filtered_df, use_container_width=True, height=260)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average Rating", round(filtered_df['Rating'].mean(), 2))
with col2:
    st.metric("Total Participants", len(filtered_df))
with col3:
    st.metric("Amount Collected", round(filtered_df['Amount Paid'].sum(), 2))
