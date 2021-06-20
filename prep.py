import pandas as pd
from sklearn.model_selection import train_test_split


class PreProcess:
    def __init__(self, random_state):
        self.rs = random_state
    
    def onehot_encoder(self, data):
        return pd.get_dummies(data)

    # Prepare features and label
    def split_feature_label(self, data, col_label:str):
        X=data.drop(col_label,axis=1)
        y=data[col_label]
        return X, y

    # Split data for training and testing set
    def split_data(self, X, Y, test_size:float=0.3):
        X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                            test_size = test_size,
                                                            random_state = self.rs,
                                                            stratify=Y)
        return X_train, X_test, y_train, y_test