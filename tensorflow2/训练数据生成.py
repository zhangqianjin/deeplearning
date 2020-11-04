#多分类
import numpy as np
from sklearn import datasets

seed = np.random.randint(int(1e9))
#x, y = datasets.make_classification(n_samples=100, n_features= 20,n_classes=10)
x, y = datasets.make_blobs(n_samples=5000000, n_features=20, centers=10, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True)
x_list = x.tolist()
y_list = y.tolist()
for i in range(len(y_list)):
    data = []
    data.extend(x_list[i])
    data.append(y_list[i])
    data_str = list(map(str, data))
    print(",".join(data_str))
