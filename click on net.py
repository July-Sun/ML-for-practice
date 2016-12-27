#!/usr/bin/python
# -*- coding: cp936 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

path = 'C://Users//507//Desktop//data for ML practice//boxoffice//狗熊皮鞋.csv'
# pandas读入
data = pd.read_csv(path)   
x = data[['click']]
y = data['show']
#print x
#print y
print np.shape(x)
print np.shape(y)

plt.plot(x, y, 'ro', label='show on click')
plt.ylim(y.min(), y.max())
plt.legend(loc='upper left')
plt.grid()
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
linreg = LinearRegression()
model = linreg.fit(x_train, y_train)
y_hat = linreg.predict(np.array(x_test))
reg = DecisionTreeRegressor()
dt = reg.fit(x_train, y_train)
y_hat1 = dt.predict(np.array(x_test))

mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
mse1 = np.average((y_hat1 - np.array(y_test)) ** 2)  # Mean Squared Error
rmse1 = np.sqrt(mse1)  # Root Mean Squared Error
print mse, rmse,mse1, rmse1,



plt.plot(x, y, 'r*', linewidth=2, label='Actual')
plt.plot(x_test, y_hat, 'g-', linewidth=2, label='Predict_lr')
plt.plot(x_test, y_hat1, 'r-', linewidth=2, label='Predict1_RF')
plt.legend(loc='upper left')
plt.grid()
plt.show()

depth = [2, 4, 6, 8, 10]
clr = 'rgbmy'
reg = [DecisionTreeRegressor(criterion='mse', max_depth=depth[0]),
       DecisionTreeRegressor(criterion='mse', max_depth=depth[1]),
       DecisionTreeRegressor(criterion='mse', max_depth=depth[2]),
       DecisionTreeRegressor(criterion='mse', max_depth=depth[3]),
       DecisionTreeRegressor(criterion='mse', max_depth=depth[4])]

plt.plot(x, y, 'k^', linewidth=2, label='Actual')
for i, r in enumerate(reg):
    dt = r.fit(x, y)
    y_hat2 = dt.predict(x_test)
    plt.plot(x_test, y_hat2, '-', color=clr[i], linewidth=2, label='Depth=%d' % depth[i])
plt.legend(loc='upper left')
plt.grid()


t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
plt.plot(t, y_hat2, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()

