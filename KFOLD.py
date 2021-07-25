import pandas as pd 
from sklearn import model_selection

if __name__ == "__main__":

    df = pd.read_csv("/home/priyanka/Downloads/winequality-red.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=5)
    kf.split(X=df)
    print(kf)
    for fold,(trn_,val_) in enumerate(kf.split(X=df)): #iterator(kf.split(X=df)
        df.loc[val_,'kfold'] = fold
        print(df.loc[val_,'kfold'])
    df.to_csv("train_folds.csv",index=False)