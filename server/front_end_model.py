import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import pairwise_distances

# Terrible hack to import from above directory.
import sys
sys.path.insert(1, '../')

from eda import clean_data
from spotify import get_song_features
import pickle

def train_model():
    df = pd.read_csv("../spotify_songs.csv")
    df = clean_data(df)

    train, test = train_test_split(df, test_size = 0.2, random_state = 20)
    x_train = train.drop(columns = ["track_name", "track_artist", "playlist_genre"])

    x_train = train.drop(columns = ["track_name", "track_artist", "playlist_genre"])
    y_train = train["playlist_genre"]

    rf = RandomForestClassifier(n_estimators = 195, criterion = "gini", min_samples_leaf = 1, min_samples_split = 3, random_state = 20)
    rf.fit(x_train, y_train)

    with open('random_forest.pkl','wb') as file:
        pickle.dump(rf, file)

def predict(song_df):
    with open('random_forest.pkl', 'rb') as file:
        rf = pickle.load(file)

    df = pd.read_csv("../spotify_songs.csv")
    df = clean_data(df)

    scaler = MinMaxScaler(feature_range = (0,1))
    song_features = song_df.drop(columns = ["track_name", "track_artist"])
    song_features[["loudness", "tempo", "duration_ms"]] = scaler.fit_transform(song_features[["loudness", "tempo", "duration_ms"]])
    genre_prediction = rf.predict(song_features)[0]
    possible_songs = df.loc[df["playlist_genre"] == genre_prediction].reset_index(drop = True)
    possibles_songs_features = possible_songs.drop(columns = ["track_name", "track_artist", "playlist_genre"])
    distances = pairwise_distances(song_features, possibles_songs_features, metric = "euclidean")[0]
    recommendations_indices = np.argsort(distances)[0:6]
    
    recommendations = []
    for index in recommendations_indices:
        song = possible_songs.loc[index, "track_name"]
        artist = possible_songs.loc[index, "track_artist"]
        if song in song_df["track_name"] and artist in song_df["track_artist"]:
            continue
        recommendations.append(song + " by " + artist)
        if len(recommendations) == 5:
            break
    return genre_prediction, recommendations

if __name__ == "__main__":
    song = get_song_features("numb")
    print("Song:", song)
    columns = ["track_name", "track_artist", "track_popularity", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "liveness", "valence", "tempo", "duration_ms"]
    song_df = pd.DataFrame(data = [song], columns = columns)
    genre, recommendations = predict(song_df)
    print("Genre:", genre)
    print("Recommendations:", recommendations)
