import pandas as pd
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("spotify_songs.csv")

df = df.drop(columns=['playlist_id', 'playlist_name', 'playlist_subgenre'])
df.drop_duplicates(subset='track_id', inplace=True)
df.drop_duplicates(subset=['track_name', 'track_artist', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'], inplace=True)

df = df.dropna(how='any',axis=0)

scaler = MinMaxScaler(feature_range = (0,1))
df[["loudness", "tempo", "duration_ms"]] = scaler.fit_transform(df[["loudness", "tempo", "duration_ms"]])

df = df.reset_index(drop = True)

df = df.drop(columns = ["instrumentalness", "track_id", "track_name", "track_artist", "track_album_id", "track_album_name", "track_album_release_date"])

labels = df["playlist_genre"].unique()
numeric_categories = ["danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence", "tempo", "duration_ms"]
for label in labels:
    genre_group = df.loc[df['playlist_genre'] == label]
    remove_indices = []
    for category in numeric_categories:
        q75 = genre_group[category].quantile(0.75)
        q25 = genre_group[category].quantile(0.25)
        iqr = q75 - q25
        remove_indices = genre_group.loc[genre_group[category] >= q75 + 1.5 * iqr].index.tolist()
        remove_indices.extend(genre_group.loc[genre_group[category] <= q25 - 1.5 * iqr].index.tolist())
    df = df.drop(remove_indices)
    df = df.reset_index(drop = True)

x = df.drop(columns = ["playlist_genre"])
y = df["playlist_genre"]

train, test = train_test_split(df, test_size = 0.2, random_state = 20)
x_train, y_train = train.drop(columns = ["playlist_genre"]), train["playlist_genre"]
x_test, y_test = test.drop(columns = ["playlist_genre"]), test["playlist_genre"]

# Grid Search!
degree = [5, 7]
kernels = ["rbf", "poly"]
regularization = [1, 3, 5]
gamma = [0.1, 0.2, 0.5]

from sklearn.svm import SVC

param_grid = dict(degree = degree,
                  kernel = kernels,
                  C = regularization,
                  gamma = gamma)
clf = SVC(random_state = 20)
grid = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 3)
grid.fit(x_train,y_train)

print("Optimal Hyper-parameters:", grid.best_params_)
degree = grid.best_params_["degree"]
kernels = grid.best_params_["kernel"]
regularization = grid.best_params_["C"]
gamma = grid.best_params_["gamma"]

# Optimal parameters: kernel: poly, degree: 7

from sklearn.preprocessing import StandardScaler

svc_rbf = SVC(kernel='poly', degree=7)

scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Training...")
svc_rbf.fit(x_train_scaled, np.asarray(y_train))

print("Classifying...")
y_pred = svc_rbf.predict(x_test_scaled)
print("Done!")
print('RBF Kernel')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average = "macro"))
print(classification_report(y_test, y_pred))

cross_validation = cross_validate(svc_rbf, x, y, cv = 10, scoring = ["accuracy"])
average_accuracy = sum(cross_validation["test_accuracy"]) / len(cross_validation["test_accuracy"])
print("Accuracy values:", cross_validation["test_accuracy"])
print("Average accuracy:", average_accuracy, "\n")
