#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

if __name__ == "__main__":
    path = 'C://Users//507//Desktop//data for ML practice//boxoffice//boxoffice.csv'
    # # 手写读取数据 - 请自行分析，在8.2.Iris代码中给出类似的例子
    # f = file(path)
    # x = []
    # y = []
    # for i, d in enumerate(f):
    #     if i == 0:
    #         continue
    #     d = d.strip()
    #     if not d:
    #         continue
    #     d = map(float, d.split(','))
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    # print x
    # print y
    # x = np.array(x)
    # y = np.array(y)

    # # Python自带库
    # f = file(path, 'rb')
    # print f
    # d = csv.reader(f)
    # for line in d:
    #     print line
    # f.close()

    # # numpy读入
    #p = np.loadtxt(path, delimiter=',', skiprows=1)
    #print p

    # pandas读入
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    x = data[['week', 'runtime']]
    # x = data[['TV', 'Radio']]
    y = data['boxoffice']
    #print x
    #print y
    print np.shape(x)
    print np.shape(y)


    # # 绘制1
    plt.plot(data['week'], y, 'ro-', label='week')
    plt.plot(data['runtime'], y, 'g^-', label='runtime')
    #plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    # #
    # # 绘制2


    plt.figure(figsize=(9,12))
    plt.subplot(211)
    plt.plot(data['week'], y, 'ro')
    plt.title('week')
    plt.grid()
    plt.subplot(212)
    plt.plot(data['runtime'], y, 'g^')
    plt.title('runtime')
    plt.grid()
    #plt.subplot(313)
    #plt.plot(data['Newspaper'], y, 'b*')
    #plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # print x_train, y_train
    linreg = LinearRegression()
    model1 = linreg.fit(x_train, y_train)
    svr = SVR(kernel="linear")
    model2 = svr.fit(x_train, y_train)
    print model1
    print linreg.coef_
    print linreg.intercept_
    print model2
    print svr.coef_
    print svr.intercept_

    y_hat1 = linreg.predict(np.array(x_test))
    y_hat2 = svr.predict(np.array(x_test))
    mse1 = np.average((y_hat1 - np.array(y_test)) ** 2)
    mse2 = np.average((y_hat2 - np.array(y_test)) ** 2) # Mean Squared Error
    rmse1 = np.sqrt(mse1)
    rmse2 = np.sqrt(mse2)# Root Mean Squared Error
    print mse1
    print mse2

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat1, 'g-', linewidth=2, label='Predict_linear')
    plt.plot(t, y_hat2, 'b-', linewidth=2, label='Predict_svr')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
