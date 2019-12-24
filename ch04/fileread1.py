import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter=',',dtype=np.float)
x_data = xy[:,:3] # == x_data = xy[:,:-1]
y_data = xy[:,3:4] # == y_data = xy[:,[-1]]

print(x_data, len(x_data), x_data.shape)
print(y_data,len(y_data), y_data.shape)

