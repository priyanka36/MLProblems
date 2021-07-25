import numpy as np 
import pandas as pd 

from sklearn import datasets
from sklearn import model_selection

def create_folds(data):
    data["kfold"]=-1
    data = data.sample(frac=1).reset_index(drop=True)
    num_bins = int(np.floor(1+np.log2(len(data))))
    print(num_bins)
    data.loc[:,"bins"]=pd.cut(data["target"],bins=num_bins,labels=False)
    #Pandas cut() function is used to separate the array elements into different bins . The cut function is mainly used to perform statistical analysis on scalar data.  
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f,(t_,v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_,'kfold'] = f
    data = data.drop("bins", axis=1)
    return data 

if __name__ == "__main__":
    X,y = datasets.make_regression(n_samples=15000,n_features=100,n_targets=1)
    df = pd.DataFrame(X,columns=[f"f_{i}" for i in range(X.shape[1])])
    df.loc[:,"target"] = y
    df = create_folds(df)
    print(df)




