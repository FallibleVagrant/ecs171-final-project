import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score
from eda import clean_data

df = pd.read_csv("spotify_songs.csv")

df = df.drop(columns=['playlist_id', 'playlist_name', 'playlist_subgenre'])
df.drop_duplicates(subset='track_id', inplace=True)
df.drop_duplicates(subset=['track_name', 'track_artist', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'], inplace=True)

df = df.dropna(how='any',axis=0)

scaler = StandardScaler()
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

# Preprocess with PCA
numeric_categories = ["danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence", "tempo", "duration_ms", "track_popularity", "key", "mode"]
pca = PCA(n_components=4, svd_solver='auto')
scaled_numerics = pd.DataFrame(pca.fit_transform(df[numeric_categories]))
df.drop(columns = numeric_categories, inplace=True)
df = pd.concat([df, scaled_numerics], axis = 1)

#To obviate an obscure error.
df.columns = df.columns.astype(str)

x = df.drop(columns = ["playlist_genre"])
y = df["playlist_genre"]

# Intervene and set y and df to use one hot encoding.
#y_onehot = pd.get_dummies(y, prefix="genre")
#df.drop(columns = ["playlist_genre"], inplace = True)
#df = pd.concat([df, y_onehot], axis = 1)
#y = y_onehot

cat = ["playlist_genre"]
#cats = ["genre_edm", "genre_latin", "genre_pop", "genre_r&b", "genre_rap", "genre_rock"]

train, test = train_test_split(df, test_size = 0.2, random_state = 20)
x_train = train.drop(columns = cat)
y_train = np.ravel(train[cat])

x_test = test.drop(columns = cat)
y_test = np.ravel(test[cat])

#     Uncomment this block and remove hardcoded values to run grid search.
#     (Thank you, Kevin.)
#number_estimators = [125, 195]
#min_samples_for_split = [2, 3, 4]
#min_samples_for_leaf = [1, 2, 7, 10]
#
#param_grid = dict(min_samples_split = min_samples_for_split,
#                  min_samples_leaf = min_samples_for_leaf,
#                  n_estimators = number_estimators)
#rf = RandomForestClassifier(random_state = 20)
#grid = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 10)
#grid.fit(x_train,y_train)
#
#print("Optimal Hyper-parameters:", grid.best_params_)
#optimal_min_samples_split = grid.best_params_["min_samples_split"]
#optimal_min_samples_leaf = grid.best_params_["min_samples_leaf"]
#optimal_number_estimators = grid.best_params_["n_estimators"]

# Hardcoded values!
print("WARNING! Using hardcoded optimal values instead of performing grid search.")
optimal_min_samples_split = 2
optimal_min_samples_leaf = 7
optimal_number_estimators = 125
 
optimal_rf = RandomForestClassifier(n_estimators = optimal_number_estimators, min_samples_leaf = optimal_min_samples_leaf, min_samples_split = optimal_min_samples_split, criterion = "gini", max_features = len(cat))
#Not sure what the 1-dimensional y value error is.
optimal_rf.fit(x_train, y_train)
y_pred = optimal_rf.predict(x_test)
print(y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average = "macro"))
print(classification_report(y_test, y_pred))
cross_validation = cross_validate(optimal_rf, x, y, cv = 10, scoring = ["accuracy"])
average_accuracy = sum(cross_validation["test_accuracy"]) / len(cross_validation["test_accuracy"])
print("Accuracy values:", cross_validation["test_accuracy"])
print("Average accuracy:", average_accuracy, "\n")
