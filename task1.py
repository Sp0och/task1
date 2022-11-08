############# load packages #############

import numpy as np
import pandas as pd

from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

#ExtraTrees: 0.635 (0.051)
#SVR: 0.648 (0.043)


############# read training and test data from the files #############

X_train = pd.read_csv("X_train.csv", index_col=0).to_numpy()
y_train = pd.read_csv("y_train.csv", index_col=0).to_numpy()
X_test = pd.read_csv("X_test.csv", index_col=0).to_numpy()

#################### preprocessing ####################

varThreshold = VarianceThreshold(threshold=0.04) # 0.0144
X_train = varThreshold.fit_transform(X_train)
X_test = varThreshold.transform(X_test)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

############ Imputation #############

Imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_train_imp = Imputer.fit_transform(X_train)

############# Outlier detection: IsolationForest #############
outlier_det = IsolationForest(random_state=0).fit(X_train_imp)
anomaly_score = outlier_det.decision_function(X_train_imp)

### to determine the contamination parameter, I propose to first plot the anomaly score function and then decide on the
### the contamination ratio based on visual thresholding.
### HERE WE NEED TO VISUALLY DETERMINE A THRESHOLD:
anomaly_threshold = 0.02
plt.plot(anomaly_score)
plt.plot(np.ones(anomaly_score.shape) * anomaly_threshold)
plt.ylabel('anomaly score')
plt.show()

mask = anomaly_score >= anomaly_threshold
X_train_imp_out = X_train_imp[mask,:]
y_train_out = y_train[mask]

print('Size of training set after initial preprocessing:', X_train_imp_out.shape)

############# Feature selection using ExtraTreesRegressor #############

selector = ExtraTreesRegressor(random_state=0)
selector = selector.fit(X_train_imp_out, y_train_out.ravel())
selector.feature_importances_

support = np.where(selector.feature_importances_ > 1.8e-3)[0]
print('number of selected features:', len(support))

X_train = X_train[:,support]
X_train = X_train[mask, :]
X_test = X_test[:,support]

############# MICE imputation #############

imp_mean = IterativeImputer(random_state=0, n_nearest_features = None, sample_posterior = False, max_iter=100) 
X_train_imp_out = imp_mean.fit_transform(X_train)
X_test_imp = imp_mean.fit_transform(X_test)


################### Second round of outlier detection #######################

outlierdet = IsolationForest(random_state=0).fit(X_train_imp_out)
anomaly_score2 = outlierdet.decision_function(X_train_imp_out)

anomaly_threshold2 = 0.0
plt.plot(anomaly_score2)
plt.plot(np.ones(anomaly_score2.shape)*anomaly_threshold2)
plt.ylabel('anomaly score after feature sel')
plt.show()

mask = anomaly_score2 >= anomaly_threshold2
X_train_imp_out = X_train_imp_out[mask,:]
y_train_out = y_train_out[mask]

print('Shape of the training set:', X_train_imp_out.shape)



############# Regression using a decision tree #############

X_train = X_train_imp_out
X_test = X_test_imp
y_train = y_train_out
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
run_list = [1,0,1,0,1]

############# Regression using a random forest #############
if run_list[0]:
    rf_reg = ExtraTreesRegressor(n_estimators=100, random_state=0, min_samples_split=5, max_samples=None)
    n_scores_0 = cross_val_score(rf_reg, X_train, y_train.ravel(), scoring='r2', cv=cv, n_jobs=-1)
    print('ExtraTrees: %.3f (%.3f)' % (np.mean(n_scores_0), np.std(n_scores_0)))
    regressor = rf_reg.fit(X_train, y_train.ravel())
    y_predicted_rf = regressor.predict(X_test)

############# Regression using HistGradientBoostingRegressor #############
if run_list[1]:
    HGB_reg = HistGradientBoostingRegressor(max_iter=200, learning_rate = 0.1, l2_regularization = 10)
    n_scores_1 = cross_val_score(HGB_reg, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
    print('HBG: %.3f (%.3f)' % (np.mean(n_scores_1), np.std(n_scores_1)))
    regressor = HGB_reg.fit(X_train, y_train)
    y_predicted_hgb = regressor.predict(X_test)

############# Regression using SVR #############    
if run_list[2]:
    SVR_reg = SVR(kernel='rbf', C=100)
    n_scores_2 = cross_val_score(SVR_reg, X_train, y_train.ravel(), scoring='r2', cv=cv, n_jobs=-1)
    print('SVR: %.3f (%.3f)' % (np.mean(n_scores_2), np.std(n_scores_2)))
    regressor = SVR_reg.fit(X_train, y_train.ravel())
    y_predicted_svr = regressor.predict(X_test)
    
############# Regression using adaboost #############
if run_list[3]:
    ada_regr = AdaBoostRegressor(random_state=0, n_estimators=1000, loss='square', learning_rate=0.5)
    n_scores_3 = cross_val_score(ada_regr, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
    print('Ada: %.3f (%.3f)' % (np.mean(n_scores_3), np.std(n_scores_3)))

############# Regression using MLP #############
if run_list[4]:
    MLP_reg = MLPRegressor(random_state=0, max_iter=10000,activation='tanh',solver='sgd',alpha=10, hidden_layer_sizes=(200))
    n_scores_4 = cross_val_score(MLP_reg, X_train, y_train.ravel(), scoring='r2', cv=cv, n_jobs=-1)
    print('MLP: %.3f (%.3f)' % (np.mean(n_scores_4), np.std(n_scores_4)))
    regressor = MLP_reg.fit(X_train, y_train.ravel())
    y_predicted_mlp = regressor.predict(X_test)

if False:
    reg_model = DecisionTreeRegressor(random_state=0, min_samples_split=0.01)
    n_scores = cross_val_score(reg_model, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
    print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

############# Write out the predictions to a csv file #############
y_predicted = (1/3)*(y_predicted_svr + y_predicted_mlp + y_predicted_rf)
y_predicted_df = pd.DataFrame(y_predicted, columns=['y'])
y_predicted_df.to_csv("thesmartones_submission_1.csv", index_label='id')
