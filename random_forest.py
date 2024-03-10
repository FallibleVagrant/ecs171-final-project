import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from eda import clean_data

def test_model(df):
    train, test = train_test_split(df, test_size = 0.2, random_state = 20)
    x_train = train.drop(columns = ["track_name", "track_artist", "playlist_genre"])
    y_train = train["playlist_genre"]
    
    x_test = test.drop(columns = ["track_name", "track_artist", "playlist_genre"])
    y_test = test["playlist_genre"]

    number_estimators = [125, 155, 195]
    min_samples_for_split = [3, 4, 5]
    min_samples_for_leaf = [1, 2, 3]

    param_grid = dict(n_estimators = number_estimators, min_samples_split = min_samples_for_split,
                      min_samples_leaf = min_samples_for_leaf)
    rf = RandomForestClassifier(random_state = 20)
    grid = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3)
    grid.fit(x_train,y_train)

    print("Optimal Hyper-parameters:", grid.best_params_)
    optimal_n_estimators = grid.best_params_["n_estimators"]
    optimal_min_samples_split = grid.best_params_["min_samples_split"]
    optimal_min_samples_leaf = grid.best_params_["min_samples_leaf"]

    optimal_rf = RandomForestClassifier(n_estimators = optimal_n_estimators, min_samples_split = optimal_min_samples_split, min_samples_leaf = optimal_min_samples_leaf, random_state = 20)
    optimal_rf.fit(x_train, y_train)
    y_pred = optimal_rf.predict(x_test)

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average = "macro"))
    print("Recall", recall_score(y_test, y_pred, average = "macro"), "\n")

    x = df.drop(columns = ["track_name", "track_artist", "playlist_genre"])
    y = df["playlist_genre"]
    cross_validation = cross_validate(optimal_rf, x, y, cv = 10, scoring = ["accuracy"])
    average_accuracy = sum(cross_validation["test_accuracy"]) / len(cross_validation["test_accuracy"])
    print("10-fold cross validation:")
    print("Accuracy values:", cross_validation["test_accuracy"])
    print("Average accuracy:", average_accuracy)    

if __name__ == "__main__":
    df = pd.read_csv("spotify_songs.csv")
    df = clean_data(df)
    test_model(df)