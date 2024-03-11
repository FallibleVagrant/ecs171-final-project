import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from eda import clean_data

def test_model(df):
    # Ordinal encode y for DecisionTreeClassifier
    cat = ["playlist_genre"]
    enc = OrdinalEncoder()
    enc.fit(df[cat])
    df[cat] = enc.transform(df[cat])

    #x = df.drop(columns = ["playlist_genre"])
    #y = df["playlist_genre"]

    # Intervene and set y and df to use one hot encoding.
    y_onehot = pd.get_dummies(df[cat], prefix="genre")
    df.drop(columns = ["playlist_genre"], inplace = True)
    df = pd.concat([df, y_onehot], axis = 1)

    #cats = ["genre_edm", "genre_latin", "genre_pop", "genre_r&b", "genre_rap", "genre_rock"]
    cats = ["playlist_genre"]

    train, test = train_test_split(df.drop(columns = ["track_name", "track_artist"]), test_size = 0.2, random_state = 20)
    x_train = train.drop(columns = cats)
    y_train = train[cats]

    x_test = test.drop(columns = cats)
    y_test = test[cats]

#     Uncomment this block and remove hardcoded values to run grid search.
#     (Thank you, Kevin.)
#    min_samples_for_split = [2, 3, 4]
#    min_samples_for_leaf = [1, 3, 5, 7, 10]
#    
#    param_grid = dict(min_samples_split = min_samples_for_split,
#                      min_samples_leaf = min_samples_for_leaf)
#    clf = DecisionTreeClassifier(random_state = 20)
#    grid = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 10)
#    grid.fit(x_train,y_train)
#    
#    print("Optimal Hyper-parameters:", grid.best_params_)
#    optimal_min_samples_split = grid.best_params_["min_samples_split"]
#    optimal_min_samples_leaf = grid.best_params_["min_samples_leaf"]

    # Hardcoded values!
    optimal_min_samples_split = 2
    optimal_min_samples_leaf = 7
    
    clf = DecisionTreeClassifier(criterion = "entropy", max_features = len(cats), min_samples_split=optimal_min_samples_split, min_samples_leaf=optimal_min_samples_leaf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average = "macro"))
    print("Recall", recall_score(y_test, y_pred, average = "macro"), "\n")

    x = df.drop(columns = ["track_name", "track_artist", "playlist_genre"])
    y = df["playlist_genre"]
    cross_validation = cross_validate(clf, x, y, cv = 10, scoring = ["accuracy"])
    average_accuracy = sum(cross_validation["test_accuracy"]) / len(cross_validation["test_accuracy"])
    print("10-fold cross validation:")
    print("Accuracy values:", cross_validation["test_accuracy"])
    print("Average accuracy:", average_accuracy)

if __name__ == "__main__":
    df = pd.read_csv("spotify_songs.csv")
    df = clean_data(df)
    test_model(df)
