# Import libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from prep import PreProcess
from model import Classifier
from eval import Report

if __name__ == '__main__':
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.3

    # Read data
    df = pd.read_csv("./WA_Fn-UseC_-HR-Employee-Attrition.csv")
    remove_col = ["EmployeeCount","EmployeeNumber","Over18","StandardHours"]
    df.drop(remove_col, axis=1, inplace=True)

    # LabelEncoder
    label_enc = LabelEncoder()
    label_attribute = "Attrition"
    df[label_attribute] = label_enc.fit_transform(df[[label_attribute]])

    prep_obj = PreProcess(random_state=RANDOM_STATE)

    # One-hot encoder
    df_preprocess = prep_obj.onehot_encoder(df)

    X, y = prep_obj.split_feature_label(df_preprocess, label_attribute)

    X_train, X_test, y_train, y_test = prep_obj.split_data(X=X, y=y, test_size=TEST_SIZE)

    clf = Classifier(random_state=RANDOM_STATE)
    rpt = Report()

    # Decision Tree
    print('\n------ Decision Tree ------')
    dtree_clf = clf.fit(X=X_train, y=y_train, classifier="dtree")
    y_pred = clf.predict(cfl=dtree_clf, test_data=X_test)
    
    print('\n-- Test Confusion Martix --')
    rpt.confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    print('\n-- Test Classification Report --')
    rpt.classification_report(y_true=y_test, y_pred=y_pred)

    # Random Forest
    print('\n------ Random Forest ------')
    rf_clf = clf.fit(X=X_train, y=y_train, classifier="random_forest")
    y_pred = clf.predict(cfl=rf_clf, test_data=X_test)
    
    print('\n-- Test Confusion Martix --')
    rpt.confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    print('\n-- Test Classification Report --')
    rpt.classification_report(y_true=y_test, y_pred=y_pred)

