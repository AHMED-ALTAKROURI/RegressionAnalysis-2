from pandas import read_csv
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge


"""
Plotting a different model with one feature component.
Training a different regression models using sklearn.
"""




# load the data set from CSV file
dataset = read_csv('C://Users/Ahmed/Desktop/backup/MyProject/ML-EXP/Regresstion/dataset_1.csv', header=None)
# replace missing data with zeros.
dataset.fillna(0, inplace=True)


# extracting feature ID 21, change the features you want to use to fit the model here:
# features = dataset[[2,,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]

features = dataset[[21]]

# target value
targets = dataset[23]



""" Plotting simple feautre with the regression model"""
# training set
training_X = features.values
training_Y = targets.values


# Regression models
clf = GradientBoostingRegressor()
r = linear_model.LinearRegression()
ridge = Ridge(alpha=1.0)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1, verbose=True)

y_rbf = ridge.fit(training_X, training_Y).predict(training_X)

lw = 2
plt.scatter(training_X, training_Y, color='red', label='data')
plt.plot(training_X, y_rbf, color='blue', lw=lw, label='Ridge Model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Ridge Model')
plt.legend()
plt.show()



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls', 'verbose': 1}
clf = GradientBoostingRegressor(**params)
r = linear_model.LinearRegression()
ridge = Ridge(alpha=1.0)



# test the model with 10 fold cross validation
predicted = cross_val_predict(svr_rbf,training_X,training_Y, cv=10)



# report the results using diff evaluation scores
print("Mean absolute error ",mean_absolute_error(training_Y, predicted))
print("explained_variance_score ",explained_variance_score(training_Y, predicted))
print("Mean Squared Error",mean_squared_error(training_Y, predicted))
print("R2",r2_score(training_Y, predicted))



"""
all 10 folds cross-validation

Model                                Explained variance score         Mean absolute error      Mean squared error    R² score, the coefficient of determination

--------Simple Linear Regression 
Mean absolute error  0.07042462538734469
explained_variance_score  0.0004416111465148642
Mean Squared Error 0.02868278898950552
R2 0.0004416110718995503

----------SVR Rbf 
explained_variance_score  -0.0007286527751753091
Mean Squared Error 0.030256845295889515
R2 -0.054412231288652135

explained_variance_score  -0.0006681239841130893
Mean Squared Error 0.030258686679529447
R2 -0.05447640114553742

----------Gradient Boosting regression
Mean absolute error  0.07058640152386743
explained_variance_score  -0.002300711123233379
Mean Squared Error 0.028761481551683146
R2 -0.002300723737301613

-----------Ridge Regression
Mean absolute error  0.07042462385867472
explained_variance_score  0.00044161228873362823
Mean Squared Error 0.0286827889567291
R2 0.0004416122141157608


"""
