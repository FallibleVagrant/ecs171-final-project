import pandas as pd

df = pd.read_csv("spotify_songs.csv")

df = df.drop(columns=['playlist_id', 'playlist_name', 'playlist_genre', 'playlist_subgenre'])
df.drop_duplicates(subset='track_id', inplace=True)
df.drop_duplicates(subset='track_name', inplace=True)

df = df.dropna(how='any',axis=0)

df.to_csv("trimmed_spotify_songs.csv", index=False)
