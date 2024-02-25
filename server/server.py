from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/")
def receive_form_data():
    try:
        song_name = request.form["song_name"]
        #We have the song name; now we can do a spotify API request to figure out features?
        #...
        #Send features to our model, and display the results.
        songs = ["test_song_1", "test_song_2"]
        return render_template("index.html", output_available=True, songs=songs)
    #If song name isn't provided, a KeyError will be raised,
    #and we'll try to look for features instead.
    except KeyError as e:
        test_feature_1 = request.form["test_feature_1"] 
        test_feature_2 = request.form["test_feature_2"]
        test_feature_3 = request.form["test_feature_3"]
        features = [test_feature_1, test_feature_2, test_feature_3]

        #If there is an empty string, raise a KeyError anyways.
        for feature in features:
            if feature == "":
                return render_template("index.html", error_available=True, error="Submitted incomplete form!")

        #We have the features; send them to the model, and display the resulting songs.
        songs = ["test_song_1", "test_song_2"]
        return render_template("index.html", features_available=True, features=features, output_available=True, songs=songs)

from flask import url_for
from flask import Response

@app.get("/style.css")
def return_style():
    return Response(url_for("static", filename="style.css"), mimetype="text/css")
