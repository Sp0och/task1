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
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

#Size of training set after initial preprocessing: (1151, 766)
#Shape of the training set: (1097, 74)
#HBG: 0.648 (0.056)
#SVR: 0.678 (0.041)
#MLP: 0.661 (0.051)


############# read training and test data from the files #############

X_train = pd.read_csv("X_train.csv", index_col=0).to_numpy()
y_train = pd.read_csv("y_train.csv", index_col=0).to_numpy()
X_test = pd.read_csv("X_test.csv", index_col=0).to_numpy()

#################### preprocessing ####################

varThreshold = VarianceThreshold(threshold=0.02)
varThreshold.fit(np.concatenate((X_train, X_test)))
X_train = varThreshold.transform(X_train)
X_test = varThreshold.transform(X_test)

scaler = preprocessing.StandardScaler()
scaler.fit(np.concatenate((X_train, X_test)))
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

############ Imputation #############

Imputer = SimpleImputer(missing_values=np.nan, strategy='median')
Imputer.fit(np.concatenate((X_train, X_test)))
X_train_imp = Imputer.transform(X_train)
X_test_imp = Imputer.transform(X_test)

############# Outlier detection: IsolationForest #############
outlier_det = IsolationForest(random_state=0).fit(np.concatenate((X_train_imp, X_test_imp)))
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
support = selector.feature_importances_ > 1.7e-3
fit = SelectKBest(mutual_info_regression, k=30).fit(X_train_imp_out, y_train_out.ravel())
features = fit.get_support()
support = np.logical_or(features, support)

X_train = X_train[:,support]
X_train = X_train[mask, :]
X_test = X_test[:,support]

############# MICE imputation #############

imp_mean = IterativeImputer(random_state=0, n_nearest_features = None, sample_posterior = False, max_iter=100) 
imp_mean.fit(np.concatenate((X_train, X_test)))
X_train_imp_out = imp_mean.transform(X_train)
X_test_imp = imp_mean.transform(X_test)

################### Second round of outlier detection #######################

outlierdet = IsolationForest(random_state=0).fit(np.concatenate((X_train_imp_out, X_test_imp)))
anomaly_score2 = outlierdet.decision_function(X_train_imp_out)

anomaly_threshold2 = -0.02
plt.plot(anomaly_score2)
plt.plot(np.ones(anomaly_score2.shape)*anomaly_threshold2)
plt.ylabel('anomaly score after feature sel')
plt.show()

mask = anomaly_score2 >= anomaly_threshold2
X_train_imp_out = X_train_imp_out[mask,:]
y_train_out = y_train_out[mask]

print('Shape of the training set:', X_train_imp_out.shape)


#%%
############# Regression using a decision tree #############

X_train = X_train_imp_out
X_test = X_test_imp
y_train = y_train_out.ravel()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

############# Regression using HistGradientBoostingRegressor #############
HGB_reg = HistGradientBoostingRegressor(max_iter=200, random_state=0, l2_regularization = 10)
n_scores_1 = cross_val_score(HGB_reg, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
print('HBG: %.3f (%.3f)' % (np.mean(n_scores_1), np.std(n_scores_1)))
regressor = HGB_reg.fit(X_train, y_train)
y_predicted_hgb = regressor.predict(X_test)

############# Regression using SVR #############    
SVR_reg = SVR(kernel='rbf', C=100)
n_scores_2 = cross_val_score(SVR_reg, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
print('SVR: %.3f (%.3f)' % (np.mean(n_scores_2), np.std(n_scores_2)))
regressor = SVR_reg.fit(X_train, y_train)
y_predicted_svr = regressor.predict(X_test)
    
############# Regression using MLP #############
MLP_reg = MLPRegressor(random_state=0, max_iter=10000,activation='tanh',solver='sgd',alpha=10, hidden_layer_sizes=(200))
n_scores_4 = cross_val_score(MLP_reg, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
print('MLP: %.3f (%.3f)' % (np.mean(n_scores_4), np.std(n_scores_4)))
regressor = MLP_reg.fit(X_train, y_train)
y_predicted_mlp = regressor.predict(X_test)

############# Write out the predictions to a csv file #############
y_predicted = (1/3)*(y_predicted_svr + y_predicted_mlp + y_predicted_hgb)
y_predicted_df = pd.DataFrame(y_predicted, columns=['y'])
y_predicted_df.to_csv("submission.csv", index_label='id')
