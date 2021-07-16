
import pandas as pd
import numpy as np
np.random.seed=0
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.stats import entropy
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

def benchmark(X_train,y_train,X_test,y_test,Case):
    gnb.fit(X_train, y_train)
    sf.fit(X_train,y_train)
    dt.fit(X_train,y_train)
    clf.fit(X_train, y_train)
    logging.info("Classification Models Trained")

    y_pred_lf = clf.predict(X_test)
    y_pred_gnb = gnb.predict(X_test)
    y_pred_svb = sf.predict(X_test)
    y_pred_dt=dt.predict(X_test)
    logging.info("States Predicted")

    score_table=pd.DataFrame(columns=['Case','Method','Accuracy','Precision','Recall','F1'])
    score_table.loc[0] = [Case,'Naive Bayes',accuracy_score(y_test,y_pred_gnb),precision_score(y_test,y_pred_gnb,average='weighted'),recall_score(y_test,y_pred_gnb,average='weighted'),f1_score(y_test,y_pred_gnb,average='weighted')]
    score_table.loc[1] = [Case,'Logistic Reg',accuracy_score(y_test,y_pred_lf),precision_score(y_test,y_pred_lf,average='weighted'),recall_score(y_test,y_pred_lf,average='weighted'),f1_score(y_test,y_pred_lf,average='weighted')]
    score_table.loc[2] = [Case,'Support Vector Machines',accuracy_score(y_test,y_pred_svb),precision_score(y_test,y_pred_svb,average='weighted'),recall_score(y_test,y_pred_svb,average='weighted'),f1_score(y_test,y_pred_svb,average='weighted')]
    score_table.loc[3] = [Case,'Decision Trees',accuracy_score(y_test,y_pred_dt),precision_score(y_test,y_pred_dt,average='weighted'),recall_score(y_test,y_pred_dt,average='weighted'),f1_score(y_test,y_pred_dt,average='weighted')]

    return score_table
