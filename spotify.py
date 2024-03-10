import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def get_song(name):
    client_id = "cadac62f48aa440b957ed19a23128f45"
    client_secret = "b029904c925b495db4dc7dc388afd77d"

    client_credential = SpotifyClientCredentials(client_id = client_id, client_secret = client_secret)
    spotify = spotipy.Spotify(auth_manager = client_credential)

    results = spotify.search(q = name, limit = 1, type = "track")

    if len(results["tracks"]["items"]) > 0:
        item = results["tracks"]["items"][0]
        song = item["name"] + " by " + item["artists"][0]["name"]
        id = item["id"]
        track_popularity = item["popularity"]

        song_features = spotify.audio_features([id])[0]

        danceability = song_features["danceability"]
        energy = song_features["energy"]
        key = song_features["key"]
        loudness = song_features["loudness"]
        mode = song_features["mode"]
        speechiness = song_features["speechiness"]
        acousticness = song_features["acousticness"]
        liveness = song_features["liveness"]
        valence = song_features["valence"]
        tempo = song_features["tempo"]
        duration_ms = song_features["duration_ms"]

        return song, [track_popularity, danceability, energy, key, loudness, mode, speechiness, acousticness, liveness, valence, tempo, duration_ms]
    else:
        return None, None