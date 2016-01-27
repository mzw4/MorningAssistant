import os
import numpy as np
import matplotlib.pyplot as plt


# form training data and labels
# X = np.empty((0, 5000), int)
# y = np.empty((0, 1), int)

# for fname in os.listdir('clap_data/claps'):
#     data = np.load("clap_data/claps/%s"%fname)
#     X = np.append(X, data, axis=0)
#     y = np.append(y, [1] * data.shape[0])

# print X.shape, y.shape

# # for vec in X[:20]:
# #     plt.plot(vec)
# #     plt.show()

# for fname in os.listdir('clap_data/noclaps'):
#     data = np.load("clap_data/noclaps/%s"%fname)
#     X = np.append(X, data, axis=0)
#     y = np.append(y, [0] * data.shape[0])

# for vec in X[-20:]:
#     plt.plot(vec)
#     plt.show()




import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)


# generate points used to plot
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
print x
x = np.sort(x[:20])
y = f(x)

print x, x_plot

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

plt.plot(x_plot, f(x_plot), label="ground truth")
plt.scatter(x, y, label="training points")

for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()