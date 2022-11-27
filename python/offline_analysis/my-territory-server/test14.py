import numpy as np

from background import gzgtgs_1

a = gzgtgs_1.default.get(1)
b = np.array(a) + 1
c = gzgtgs_1.default.update({1: b})

print(a)
print(b)
print(gzgtgs_1.default.get(1))