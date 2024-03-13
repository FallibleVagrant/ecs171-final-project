import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def graph_distributions(df):
    fig, axes = plt.subplots(7, 2, figsize = (50, 50))
    sns.histplot(df, x = "danceability", hue = "playlist_genre", ax = axes[0][0])
    sns.histplot(df, x = "energy", hue = "playlist_genre", ax = axes[0][1])
    sns.histplot(df, x = "loudness", hue = "playlist_genre", ax = axes[1][0])
    sns.histplot(df, x = "speechiness", hue = "playlist_genre", ax = axes[1][1])
    sns.histplot(df, x = "acousticness", hue = "playlist_genre", ax = axes[2][0])
    sns.histplot(df, x = "instrumentalness", hue = "playlist_genre", ax = axes[2][1])
    sns.histplot(df, x = "liveness", hue = "playlist_genre", ax = axes[3][0])
    sns.histplot(df, x = "valence", hue = "playlist_genre", ax = axes[3][1])
    sns.histplot(df, x = "tempo", hue = "playlist_genre", ax = axes[4][0])
    sns.histplot(df, x = "duration_ms", hue = "playlist_genre", ax = axes[4][1])
    sns.histplot(df, x = "track_popularity", hue = "playlist_genre", ax = axes[5][0])
    sns.histplot(df, x = "key", hue = "playlist_genre", ax = axes[5][1])
    sns.histplot(df, x = "mode", hue = "playlist_genre", ax = axes[6][0])
    
def graph_boxplots(df):
    fig, axes = plt.subplots(7, 2, figsize = (50, 50))
    sns.boxplot(df, x = "danceability", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[0][0])
    sns.boxplot(df, x = "energy", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[0][1])
    sns.boxplot(df, x = "loudness", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[1][0])
    sns.boxplot(df, x = "speechiness", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[1][1])
    sns.boxplot(df, x = "acousticness", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[2][0])
    sns.boxplot(df, x = "instrumentalness", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[2][1])
    sns.boxplot(df, x = "liveness", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[3][0])
    sns.boxplot(df, x = "valence", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[3][1])
    sns.boxplot(df, x = "tempo", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[4][0])
    sns.boxplot(df, x = "duration_ms", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[4][1])
    sns.boxplot(df, x = "track_popularity", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[5][0])
    sns.boxplot(df, x = "key", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[5][1])
    sns.boxplot(df, x = "mode", hue = "playlist_genre", whis = (0.20, 0.80), ax = axes[6][0])

def clean_data(df):
    df = df.drop(columns=['playlist_id', 'playlist_name', 'playlist_subgenre'])
    df.drop_duplicates(subset='track_id', inplace=True)
    df.drop_duplicates(subset=['track_name', 'track_artist', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'], inplace=True)

    df = df.dropna(how='any',axis=0)

    df = df.reset_index(drop = True)

    scaler = MinMaxScaler(feature_range = (0,1))
    df[["loudness", "tempo", "duration_ms"]] = scaler.fit_transform(df[["loudness", "tempo", "duration_ms"]])

    df = df.drop(columns = ["instrumentalness", "track_id", "track_album_id", "track_album_name", "track_album_release_date"])

    # Remove outliers for each genre
    labels = df["playlist_genre"].unique()
    numeric_categories = ["danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence", "tempo", "duration_ms"]
    for label in labels:
        genre_group = df.loc[df['playlist_genre'] == label]
        remove_indices = []
        for category in numeric_categories:
            q80 = genre_group[category].quantile(0.80)
            q20 = genre_group[category].quantile(0.20)
            iqr = q80 - q20
            remove_indices = genre_group.loc[genre_group[category] >= q80 + 1.5 * iqr].index.tolist()
            remove_indices.extend(genre_group.loc[genre_group[category] <= q20 - 1.5 * iqr].index.tolist())
        df = df.drop(remove_indices)
        df = df.reset_index(drop = True)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("spotify_songs.csv")
    graph_distributions(df)
    graph_boxplots(df)
