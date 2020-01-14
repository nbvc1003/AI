from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
dataset = loadtxt('data-03-diabetes.csv', delimiter=',')
X = dataset[:, :-1]
y = dataset[:,[-1]]
model = XGBClassifier()
model.fit(X,y)
plot_importance(model)
plt.show()
