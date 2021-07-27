import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
 
df= pd.DataFrame({'number': np.random.randint(1, 100, 10)})
print(df)
df['bins'] = pd.cut(x=df['number'],right=True, bins=[1, 20, 40, 60,
                                          80, 100])
print(df)
 
# We can check the frequency of each bin
print(df['bins'].unique())


plt.hist(n_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k') #calculating histogram