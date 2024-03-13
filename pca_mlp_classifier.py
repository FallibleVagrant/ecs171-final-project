import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from eda import clean_data

def test_model(df):
    # Preprocess with PCA
    df = df.drop(columns = ["track_name", "track_artist"])
    numeric_categories = ["danceability", "energy", "loudness", "speechiness", "acousticness", "liveness", "valence", "tempo", "duration_ms", "track_popularity", "key", "mode"]
    pca = PCA(n_components=4, svd_solver='auto')
    scaled_numerics = pd.DataFrame(pca.fit_transform(df[numeric_categories]))
    df.drop(columns = numeric_categories, inplace=True)
    df = pd.concat([df, scaled_numerics], axis = 1)

    #To obviate an obscure error.
    df.columns = df.columns.astype(str)

    train, test = train_test_split(df, test_size = 0.2, random_state = 20)
    x_train = train.drop(columns = ["playlist_genre"])
    y_train = train["playlist_genre"]
    
    x_test = test.drop(columns = ["playlist_genre"])
    y_test = test["playlist_genre"]

# Comment this block and uncomment hardcoded values to avoid running grid search.
    nodes = [(8, 14, 7), (20, 23, 37), (80, 10, 30)]
    rates = [0.02, 0.1, 0.5]
    epochs = [800, 1000, 1200]

    param_grid = dict(hidden_layer_sizes = nodes, learning_rate_init =  rates, max_iter = epochs)
    mlp = MLPClassifier(hidden_layer_sizes = (1, 3, 5), learning_rate_init = 0.2, max_iter = 250, activation = "logistic", solver = "sgd", random_state = 20, batch_size = 100)
    grid = GridSearchCV(estimator = mlp, param_grid = param_grid, cv = 3)
    grid.fit(x_train, y_train)

    print("Optimal Hyper-parameters:", grid.best_params_)
    optimal_number_of_nodes = grid.best_params_["hidden_layer_sizes"]
    optimal_learning_rate = grid.best_params_["learning_rate_init"]
    optimal_number_of_epochs = grid.best_params_["max_iter"]

    # Hardcoded values!
    #print("WARNING! Using hardcoded optimal values instead of performing grid search.")
    #optimal_number_of_nodes = (20, 23, 37)
    #optimal_learning_rate = 0.02
    #optimal_number_of_epochs = 800

    optimal_mlp = MLPClassifier(hidden_layer_sizes = optimal_number_of_nodes, learning_rate_init = optimal_learning_rate, max_iter = optimal_number_of_epochs, activation = "logistic", solver = "sgd", random_state = 20, batch_size = 100)
    optimal_mlp.fit(x_train, y_train)
    y_pred = optimal_mlp.predict(x_test)

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average = "macro"))
    print("Recall:", recall_score(y_test, y_pred, average = "macro"), "\n")

    x = df.drop(columns = ["playlist_genre"])
    y = df["playlist_genre"]
    cross_validation = cross_validate(optimal_mlp, x, y, cv = 10, scoring = ["accuracy", "precision_macro", "recall_macro"])
    average_accuracy = sum(cross_validation["test_accuracy"]) / len(cross_validation["test_accuracy"])
    average_precision = sum(cross_validation["test_precision_macro"]) / len(cross_validation["test_precision_macro"])
    average_recall = sum(cross_validation["test_recall_macro"]) / len(cross_validation["test_recall_macro"])
    print("10-fold cross validation:")
    print("Accuracy values:", cross_validation["test_accuracy"])
    print("Average accuracy:", average_accuracy, "\n")
    print("Precision values:", cross_validation["test_precision_macro"])
    print("Average precision:", average_precision, "\n")
    print("Recall values:", cross_validation["test_recall_macro"])
    print("Average recall:", average_recall)      

if __name__ == "__main__":
    df = pd.read_csv("spotify_songs.csv")
    df = clean_data(df)
    test_model(df)
