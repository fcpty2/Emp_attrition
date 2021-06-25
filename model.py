# Import libraries
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class Classifier:
    def __init__(self, random_state):
        self.rs = random_state
    
    def _create_pipeline(self, clf_obj="dtree"):
        if "tree" in clf_obj.lower():
            model_obj = self._get_dtree_obj()
        elif "forest" in clf_obj.lower():
            model_obj = self._get_rf_obj()
        else:
            return None
        
        return Pipeline([("MinMaxScaler", MinMaxScaler()), 
                        ("Model", model_obj)])

    def _get_dtree_obj(self):
        return DecisionTreeClassifier(random_state=self.rs)
    
    def _get_rf_obj(self):
        return RandomForestClassifier(random_state=self.rs)

    def fit(self, X, Y, classifier="dtree"):
        model_pipeline = self._create_pipeline(classifier)
        if model_pipeline is None:
            return None
        else:
            model_pipeline.fit(X,Y)
        return model_pipeline
    
    def predict(self, clf, test_data):
        y_pred = clf.predict(test_data)
        return y_pred
