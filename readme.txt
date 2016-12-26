this file is some basis machine learning algotithm comparison based on some inline pratical data:
1. boxoffice is based on boxoffice on China for the recent years and compared the linear regression and SVM(linear kernel)
data: is on data file with name: boxoffice.csv
train data x: week,runtime
train data y:boxoffice
train process: Cross validation

method: logistic regression & SVM(linear kernel)

result:logistic is better than default SVM 
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
[  39.21993677  547.81073056]
-46309.3257196
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
  kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
[[  19.67560836  257.20647558]]
[-20660.93124535]
135214157.681
146837629.966