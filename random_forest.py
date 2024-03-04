import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score


df = pd.read_csv("spotify_songs.csv")

df = df.drop(columns=['playlist_id', 'playlist_name', 'playlist_genre'])
df.drop_duplicates(subset='track_id', inplace=True)
df.drop_duplicates(subset=['track_name', 'track_artist', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'], inplace=True)

df = df.dropna(how='any',axis=0)

scaler = MinMaxScaler(feature_range = (0,1))
df[["loudness", "tempo", "duration_ms"]] = scaler.fit_transform(df[["loudness", "tempo", "duration_ms"]])

df = df.reset_index(drop = True)

df = df.drop(columns = ["instrumentalness", "track_id", "track_name", "track_artist", "track_album_id", "track_album_name", "track_album_release_date"])

labels = df["playlist_subgenre"].unique()
numeric_categories = ["danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence", "tempo", "duration_ms"]
for label in labels:
    genre_group = df.loc[df['playlist_subgenre'] == label]
    remove_indices = []
    for category in numeric_categories:
        q75 = genre_group["loudness"].quantile(0.75)
        q25 = genre_group["loudness"].quantile(0.25)
        iqr = q75 - q25
        remove_indices = genre_group.loc[genre_group["loudness"] >= q75 + 1.5 * iqr].index.tolist()
        remove_indices.extend(genre_group.loc[genre_group["loudness"] <= q25 - 1.5 * iqr].index.tolist())
    df = df.drop(remove_indices)
    df = df.reset_index(drop = True)

x = df.drop(columns = ["playlist_subgenre"])
y = df["playlist_subgenre"]

train, test = train_test_split(df, test_size = 0.2, random_state = 20)
x_train = train.drop(columns = ["playlist_subgenre"])
y_train = train["playlist_subgenre"]

x_test = test.drop(columns = ["playlist_subgenre"])
y_test = test["playlist_subgenre"]

# Average accuracy: 0.2707

rf = RandomForestClassifier()

rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
cross_validation = cross_validate(rf, x, y, cv = 10, scoring = ["accuracy"])
average_accuracy = sum(cross_validation["test_accuracy"]) / len(cross_validation["test_accuracy"])
print("Accuracy values:", cross_validation["test_accuracy"])
print("Average accuracy:", average_accuracy, "\n")
