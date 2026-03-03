import numpy as np
import pandas as pd

matrix = np.loadtxt("rtMatrix.txt")

# print(matrix)
print(matrix.shape)

df = pd.DataFrame(matrix)

print(df)