import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


train = pd.read_csv('train.csv')
# print(train.columns)
# print(max(train['Id']))
N = max(train['Id'])+1

corrmat = train.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

X = train[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']].to_numpy()
y = train['y'].to_numpy()

kf = KFold(n_splits=10)
coe = reg.coef_.T*0
L_buf = []
for train_index, val_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", val_index)
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    reg = LinearRegression().fit(X_train, y_train)
    L_val = 1/len(y_val)*np.sqrt(np.sum((X_val@reg.coef_.T-y_val)**2))
    L_buf.append(L_val)
    coe += reg.coef_.T
    
coe = coe/10
L_ave = sum(L_buf)/10
print(L_ave)
print(L_buf)
print(coe)
f, ax = plt.subplots(figsize=(12,3))
ax.plot(range(10),L_buf)

# test inference
test = pd.read_csv('test.csv')
X_test = test[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
y_hat_test = X_test@reg.coef_.T
# print(y_hat_test)
df_out = pd.DataFrame({'Id': range(N,N+len(y_hat_test)), 'y': y_hat_test})
df_out.to_csv('submit.csv', index=False) 
