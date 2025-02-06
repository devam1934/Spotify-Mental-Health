import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Spotify Insights", layout="wide")

# Spotify API Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id='8fb41c10ec9348fd847cf2e4569bfac3',
    client_secret='cc3facc7b8b549da9235e1770d94ad00',
    redirect_uri='https://spotify-mental-health.streamlit.app/',
    scope='user-top-read user-read-recently-played'))

# VADER Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

# Fetch recently played tracks
def get_recent_tracks():
    results = sp.current_user_recently_played(limit=50)
    track_data = []
    
    for item in results['items']:
        track = item['track']
        track_data.append({
            'track_name': track['name'],
            'artist': track['artists'][0]['name'],
            'played_at': item['played_at'],
            'duration_ms': track['duration_ms'],
            'popularity': track['popularity'],
            'genre': sp.artist(track['artists'][0]['id'])['genres'][0] if sp.artist(track['artists'][0]['id'])['genres'] else 'Unknown'
        })
    
    return pd.DataFrame(track_data)

# Feature Engineering
def extract_features(df):
    df['played_at'] = pd.to_datetime(df['played_at'])
    df['hour'] = df['played_at'].dt.hour
    df['day_of_week'] = df['played_at'].dt.dayofweek
    df['date'] = df['played_at'].dt.date
    df['listening_duration'] = df['duration_ms'] / 1000  # Convert to seconds
    df['night_listening'] = df['hour'].apply(lambda x: 1 if x >= 22 or x < 6 else 0)
    df['repeated_songs'] = df['track_name'].duplicated().astype(int)
    df['session_length'] = df.groupby(df['date'])['track_name'].transform('count')
    return df

# Aggregate Insights
def get_listening_insights(df):
    daily_listening = df.groupby('date')['listening_duration'].sum().reset_index()
    avg_listening_time = daily_listening['listening_duration'].mean()
    top_genres = df['genre'].value_counts().reset_index()
    top_genres.columns = ['genre', 'count']
    return daily_listening, avg_listening_time, top_genres

# Anomaly Detection
def detect_anomalies(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    data['anomaly_score'] = model.fit_predict(data[['hour', 'day_of_week', 'listening_duration', 'night_listening', 'repeated_songs', 'popularity', 'session_length']])
    data['anomaly_score_continuous'] = model.decision_function(data[['hour', 'day_of_week', 'listening_duration', 'night_listening', 'repeated_songs', 'popularity', 'session_length']])
    return data

# Probabilistic Risk Assessment
def calculate_risk_score(data):
    risk_factors = {
        'night_listening': 0.3,
        'repeated_songs': 0.25,
        'session_length': 0.2,
        'popularity': 0.1,
        'listening_duration': 0.15
    }
    data['risk_score'] = (data['night_listening'] * risk_factors['night_listening'] +
                          data['repeated_songs'] * risk_factors['repeated_songs'] +
                          data['session_length'] * risk_factors['session_length'] +
                          (1 - data['popularity'] / 100) * risk_factors['popularity'] +
                          data['listening_duration'] / 600 * risk_factors['listening_duration'])
    return data

# Mental Health Insights
def analyze_mental_health(data):
    unusual_behavior = data[data['anomaly_score'] == -1]
    insights = []
    if len(unusual_behavior) > 10:
        insights.append("🚨 High deviations in listening behavior detected. Consider checking in on your mental well-being.")
    if data['night_listening'].sum() > 15:
        insights.append("🌙 Excessive late-night listening detected. Poor sleep can impact mental health.")
    if data['repeated_songs'].sum() > 20:
        insights.append("🔄 Repeated song plays detected. Possible emotional fixation on specific tracks.")
    if data['risk_score'].mean() > 0.5:
        insights.append("⚠️ Elevated risk score detected. Sudden changes in behavior may indicate distress.")
    if not insights:
        insights.append("✅ No significant anomalies detected in listening behavior.")
    return insights

# Function to categorize songs based on popularity and listening duration
def categorize_songs(df):
    underrated_threshold = 180  # e.g., high listening duration (in seconds)
    overrated_threshold = 60   # e.g., high popularity score
    popularity_threshold_low = 20
    popularity_threshold_high = 60

    def get_song_category(row):
        if row['popularity'] > overrated_threshold and row['listening_duration'] < underrated_threshold:
            return 'Overrated'
        elif row['popularity'] < popularity_threshold_low and row['listening_duration'] > underrated_threshold:
            return 'Underrated'
        elif popularity_threshold_low <= row['popularity'] <= popularity_threshold_high:
            return 'Perfectly Rated'
        else:
            return 'Perfectly Rated'

    df['song_category'] = df.apply(get_song_category, axis=1)
    return df

# Function to visualize the distribution of song categories
def plot_song_category_distribution(df):
    category_counts = df['song_category'].value_counts().reset_index()
    category_counts.columns = ['Song Category', 'Count']
    
    fig = px.pie(category_counts, names='Song Category', values='Count', title='Distribution of Underrated, Overrated, and Perfectly Rated Songs')
    fig.update_traces(textinfo='percent+label')
    return fig

# Generate Listening Heatmap
def plot_listening_heatmap(data):
    heatmap_data = data.pivot_table(index='day_of_week', columns='hour', values='listening_duration', aggfunc='sum')
    fig = px.imshow(heatmap_data, labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Listening Duration'},
                    title='Listening Heatmap', color_continuous_scale='Viridis')
    return fig

def plot_popularity_distribution(df):
    fig = px.histogram(df, x='popularity', nbins=20, title="Distribution of Song Popularity")
    fig.update_layout(xaxis_title="Popularity", yaxis_title="Count")
    st.plotly_chart(fig)

# Web App with Streamlit
def main():
    st.title("Spotify Listening Behavior & Mental Health Insights")

    # Check if user is authenticated
    if not sp.current_user():
        st.write("Please log in to Spotify to see your insights.")
        # Create a login button
        if st.button("Login to Spotify"):
            # Redirect to Spotify login
            auth_url = sp.auth_manager.get_authorize_url()
            st.markdown(f"[Click here to log in]({auth_url})")
            return
    
    df = get_recent_tracks()

    features = extract_features(df)
    analyzed_data = detect_anomalies(features)
    analyzed_data = calculate_risk_score(analyzed_data)
    daily_listening, avg_listening_time, top_genres = get_listening_insights(df)
    insights = analyze_mental_health(analyzed_data)

    st.subheader("Your Recent Listening Behavior")
    st.dataframe(df[['track_name', 'artist', 'played_at']])

    plot_popularity_distribution(df)

    categorized_df = categorize_songs(df)

    # Display the distribution of song categories
    fig = plot_song_category_distribution(categorized_df)
    st.plotly_chart(fig)
    
    st.subheader("Mental Health Insights")
    for insight in insights:
        st.write(insight)
    
    st.subheader("Listening Behavior Heatmap")
    st.plotly_chart(plot_listening_heatmap(features))
    
    st.subheader("Listening Behavior & Anomalies")
    fig = px.scatter(df, x='hour', y='listening_duration', color='anomaly_score_continuous', 
                     title='Listening Behavior Analysis', 
                     labels={'hour': 'Hour of the Day', 'listening_duration': 'Listening Duration (s)'},
                     color_continuous_scale='RdBu')
    st.plotly_chart(fig)

    # Display the first few records with continuous anomaly scores
    st.subheader("Song Data with Continuous Anomaly Scores")
    st.write(df[['track_name', 'artist', 'popularity', 'listening_duration', 'anomaly_score_continuous']].head(20))
    
    # Anomaly Score Distribution
    st.subheader("Anomaly Score Distribution")
    fig = px.histogram(df, x='anomaly_score_continuous', title="Anomaly Score Distribution", nbins=50)
    st.plotly_chart(fig)
    
    st.subheader("Risk Score Distribution")
    risk_fig = px.histogram(analyzed_data, x='risk_score', nbins=20, title='Risk Score Distribution',
                            labels={'risk_score': 'Risk Score'}, color_discrete_sequence=['red'])
    st.plotly_chart(risk_fig)
    
    st.subheader("Daily Listening Time")
    daily_fig = px.line(daily_listening, x='date', y='listening_duration', title='Daily Listening Duration',
                        labels={'date': 'Date', 'listening_duration': 'Listening Time (s)'})
    st.plotly_chart(daily_fig)
    
    st.subheader("Top Genres Listened")
    genre_fig = px.bar(top_genres, x='genre', y='count', title='Top Genres',
                       labels={'genre': 'Genre', 'count': 'Number of Plays'}, color='count')
    st.plotly_chart(genre_fig)

if __name__ == "__main__":
    main()