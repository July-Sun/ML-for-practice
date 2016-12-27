this file is some basis machine learning algotithm comparison based on some inline pratical data: some data is come from "¹·ÐÜ»á" and other from internet.
1. boxoffice is based on boxoffice on China for the recent years and compared the linear regression and SVM(linear kernel)
data: is on data file with name: boxoffice.csv
train data x: week,runtime
train data y: boxoffice
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

2. click on net  is based on internetclick and show on product shows and compared the linear regression and rendom forget£¬and show differn depth on Random forest
data: is on data file with name: ¹·ÐÜÆ¤Ð¬.csv
train data x: click
train data y: show
train process: Cross validation


method: linear regression & random forset

result:random forset precision is better than linear regression but less beautiful picture.by the way ,random forest result is related to the depth of the forest.
LinearRegression£º5567129.759 2359.47658581 
random forset£º4295830.89429 2072.6386309

result is not so good due to the data and feature is not good enough£¬which show that data and feature is imporant than model and algorithm.