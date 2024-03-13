from flask import Flask
from flask import render_template
from flask import request

from spotify import get_song_features

app = Flask(__name__)

def render_error(err):
    return render_template("index.html", error_available=True, error=err)

@app.get("/")
def index():
    return render_template("index.html")

# features looks like: [track_popularity, danceability, energy, key, loudness, mode, speechiness, acousticness, liveness, valence, tempo, duration_ms]
def get_genre_from_features(model, features):
    genre = f"test {features}"

    #My python version is 3.9 and I am *NOT* dealing with multiple python installs,
    #so we're using if-else statements rather than match statements.

    #Scratch the above, we're doing one model for now.
    if model == "random_forest":
        return genre
    else:
        raise ValueError

@app.post("/")
def receive_form_data():
    try:
        song_name = request.form["song_name"]

        #We have the song name; now we can do a spotify API request to figure out features.
        features = get_song_features(song_name)
        returned_song_name = features[0] + " by " + features[1]
        features = features[2:]
        if features == None:
            return render_error("Could not retrieve features for given song!")

        #Send features to our model, and display the results.
        returned_genre = get_genre_from_features("random_forest", features)

        returned_song_name = "Song found: " + returned_song_name
        returned_genre = "Predicted genre: " + returned_genre

        #Also find the closest songs to recommend.
        #TODO...

        output_lines = [returned_song_name, returned_genre]

        for line in output_lines:
            if line == "":
                return render_error("Could not retrieve any songs from given search terms!")

        return render_template("index.html", output_available=True, output_lines=output_lines)
    #If song name isn't provided, a KeyError will be raised,
    #and we'll try to look for features instead.
    except KeyError as e:
# features looks like: [track_popularity, danceability, energy, key, loudness, mode, speechiness, acousticness, liveness, valence, tempo, duration_ms]
        track_popularity = request.form["track_popularity"] 
        danceability = request.form["danceability"]
        energy = request.form["energy"]
        key = request.form["key"]
        loudness = request.form["loudness"]
        mode = request.form["mode"]
        speechiness = request.form["speechiness"]
        acousticness = request.form["acousticness"]
        liveness = request.form["liveness"]
        valence = request.form["valence"]
        tempo = request.form["tempo"]
        duration_ms = request.form["duration_ms"]
        features = [track_popularity, danceability, energy, key, loudness, mode, speechiness, acousticness, liveness, valence, tempo, duration_ms]

        #If there is an empty string, raise a KeyError anyways.
        for feature in features:
            if feature == "":
                return render_error("Submitted incomplete form!")

        #We have the features; send them to the model, and display the resulting genre.
        output_lines = ["test_1", "test_2", f"test {features}"]
        return render_template("index.html", features_available=True, features=features, output_available=True, output_lines=output_lines)

from flask import url_for
from flask import Response

@app.get("/style.css")
def return_style():
    return Response(url_for("static", filename="style.css"), mimetype="text/css")
