import pandas as pd 
from sklearn import model_selection
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


if __name__== "__main__":
    df=pd.read_csv("/home/priyanka/Downloads/winequality-red.csv")
    df["kfold"]=-1
    df = df.sample(frac=1).reset_index(drop=True)
    
    y = df.quality.values
    print(y.shape)
    print(df.shape)
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold']=f
    df.to_csv("train_folds_stratified.csv", index=False)

b = sns.countplot(x='quality', data=df)
b.set_xlabel("quality", fontsize=20)
b.set_ylabel("count", fontsize=20)
plt.show()
