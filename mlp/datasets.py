import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import one_hot
import pandas as pd
import numpy as np
import pickle


class CustomDataSet(Dataset):

    def __init__(self, train=False, preprocess_data=False):
        self._pred_variable_name = "playlist_genre"
        if preprocess_data:
            data = pd.read_csv("spotify_songs.csv")
            data = data.drop(
                [
                    "track_id",
                    "track_name",
                    "track_artist",
                    "track_popularity",
                    "track_album_id",
                    "track_album_name",
                    "track_album_release_date",
                    "playlist_name",
                    "playlist_id",
                    "playlist_subgenre",
                ],
                axis=1,
            )  # remove useless data
            # scaler = MinMaxScaler()

            data = data.dropna()  # drop na
            # for i in data.columns:  # min max scaler
            #     if i != self._pred_variable_name:   min max scaler
            #         data[i] = scaler.fit_transform(data[[i]])

            u = np.unique(data[self._pred_variable_name])

            self.int_to_genre = {i: j for i, j in enumerate(u)}
            self.genre_to_int = {j: i for i, j in enumerate(u)}

            data[self._pred_variable_name] = data[self._pred_variable_name].apply(lambda x: self.genre_to_int[x])
            train, test = train_test_split(data, train_size=0.8, random_state=43)
            data = {}
            data["train"] = train
            data["test"] = test
            data["int_to_genre"] = self.int_to_genre
            data["genre_to_int"] = self.genre_to_int
            with open("data", "wb") as file:
                pickle.dump(data, file)
            self.data = data["train"]  # create actual variable
        
        else:
        
            # with open("/root/ucd/Interesno/ecs171_w2024/ecs171-final-project/mlp/data", "rb") as file:
            
            with open("data", "rb") as file:
                data = pickle.load(file)
                self.data = data["train" if train else "test"]
            self.int_to_genre = data["int_to_genre"]
            self.genre_to_int = data["genre_to_int"]
        self.num_of_labels = len(np.unique(self.data[self._pred_variable_name]))
        self.num_of_params = len(self.data.columns)-1


    def __len__(self):
        return len(self.data)

    def get_genre_from_int(self, x):
        try:
            tmp = self.int_to_genre[x]
            return tmp
        except KeyError:
            print("Number out of bounds")
            return

    def get_int_from_genre(self, s):
        try:
            tmp = self.genre_to_int[s]
            return tmp
        except KeyError:
            print("Name out of bounds")
            return

    def __getitem__(self, index):
        tmp = self.data.iloc[index]
        img = torch.tensor(tmp[1:]).float()
        label = one_hot(torch.tensor(tmp[0]).long(), num_classes=self.num_of_labels).float()
        return img, label


def test_dataset(preprocess_data=False):
    data = CustomDataSet(preprocess_data=preprocess_data)
    variables = torch.randint(len(data), (4,))
    print(f"Size of full data:{len(data)} \n Random variables from dataset:")
    for i, j in enumerate(variables):
        tmp = data[j.item()]
        print(f"item: {tmp[0]} \n label: {tmp[1]} \n")
    print(f"Int to label: {data.int_to_genre} \n")
    print(f"Label to int: {data.genre_to_int} \n")
    d = CustomDataSet(train=True)
    print(f"Length of train dataset: {len(d)}")
    d = CustomDataSet(train=False)
    print(f"Length of test dataset {len(d)}")    



if __name__ == "__main__":
    test_dataset(preprocess_data=True)
