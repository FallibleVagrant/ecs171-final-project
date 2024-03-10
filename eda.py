from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    df = df.drop(columns=['playlist_id', 'playlist_name', 'playlist_subgenre'])
    df.drop_duplicates(subset='track_id', inplace=True)
    df.drop_duplicates(subset=['track_name', 'track_artist', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'], inplace=True)

    df = df.dropna(how='any',axis=0)

    df = df.reset_index(drop = True)

    scaler = MinMaxScaler(feature_range = (0,1))
    df[["loudness", "tempo", "duration_ms"]] = scaler.fit_transform(df[["loudness", "tempo", "duration_ms"]])

    df = df.drop(columns = ["instrumentalness", "track_id", "track_album_id", "track_album_name", "track_album_release_date"])

    labels = df["playlist_genre"].unique()
    numeric_categories = ["danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence", "tempo", "duration_ms"]
    for label in labels:
        genre_group = df.loc[df['playlist_genre'] == label]
        remove_indices = []
        for category in numeric_categories:
            q75 = genre_group[category].quantile(0.80)
            q25 = genre_group[category].quantile(0.20)
            iqr = q75 - q25
            remove_indices = genre_group.loc[genre_group[category] >= q75 + 1.5 * iqr].index.tolist()
            remove_indices.extend(genre_group.loc[genre_group[category] <= q25 - 1.5 * iqr].index.tolist())
        df = df.drop(remove_indices)
        df = df.reset_index(drop = True)
    
    return df