import pandas as pd
from sklearn import tree
from sklearn import metrics

df = pd.read_csv('/home/priyanka/Downloads/winequality-red.csv')

print(df.info())
print(df.head())
print(df["quality"])
print(df.shape)

quality_mapping = {
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
}
#df.loc[:, "quality"] = df.quality.map(quality_mapping)
df.loc[:,"quality"] = df.quality.map(quality_mapping)

df = df.sample(frac=1).reset_index(drop=True)
df2 = df.sample(frac=1)
print(df2)
print(df2.shape)
print(df2.head())
print(df.head())

print(df.shape)

df_train = df.head(1000)
df_test = df.tail(599)

clf = tree.DecisionTreeClassifier(max_depth=3)
cols = ['fixed acidity','volatile acidity','citric acid','residual sugar',
'chlorides',
'free sulfur dioxide',
'total sulfur dioxide',
'density',
'pH',
'sulphates',
'alcohol']
clf.fit(df_train[cols],df_train.quality)
train_predictions = clf.predict(df_train[cols])
test_predictions = clf.predict(df_test[cols])

train_accuracy = metrics.accuracy_score(df_train.quality, train_predictions)
test_accuracy = metrics.accuracy_score(
df_test.quality, test_predictions
)
print(test_accuracy)
print(train_accuracy)


from sklearn import tree
from sklearn import metrics

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

train_accuracies = [0.5]
test_accuracies = [0.5]

#iterate over a few depth values 
for depth in range(1,25):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    cols = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates','alcohol']
    clf.fit(df_train[cols], df_train.quality)
    train_predictions = clf.predict(df_train[cols])
    test_predictions = clf.predict(df_test[cols])
    train_accuracy = metrics.accuracy_score(
    df_train.quality, train_predictions
    )
    test_accuracy = metrics.accuracy_score(
    df_test.quality, test_predictions
    )
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

plt.figure(figsize=(20, 20))
sns.set_style("whitegrid")
plt.plot(train_accuracies, label="train accuracy")
plt.plot(test_accuracies, label="test accuracy")
plt.legend(loc="upper left", prop={'size': 15})
plt.xticks(range(0, 26, 5))
plt.xlabel("max_depth", size=20)
plt.ylabel("accuracy", size=20)
plt.show()


# We see that the best score for test data is obtained when max_depth has a value of
# 14. As we keep increasing the value of this parameter, test accuracy remains the
# same or gets worse, but the training accuracy keeps increasing. It means that our
# simple decision tree model keeps learning about the training data better and better
# with an increase in max_depth, but the performance on test data does not improve
# at all.

# This is called overfitting.
# The model fits perfectly on the training set and performs poorly when it comes to
# the test set. This means that the model will learn the training data well but will not
# generalize on unseen samples. In the dataset above, one can build a model with very
# high max_depth which will have outstanding results on training data, but that kind
# of model is not useful as it will not provide a similar result on the real-world samples
# or live data.

# One might argue that this approach isn’t overfitting as the accuracy of the test set
# more or less remains the same. Another definition of overfitting would be when the
# test loss increases as we keep improving training loss. This is very common when
# it comes to neural networks.
# Whenever we train a neural network, we must monitor loss during the training time
# for both training and test set. If we have a very large network for a dataset which is
# quite small (i.e. very less number of samples), we will observe that the loss for both
# training and test set will decrease as we keep training. However, at some point, test
# loss will reach its minima, and after that, it will start increasing even though training
# loss decreases further. We must stop training where the validation loss reaches its
# minimum value.
# This is the most common explanation of overfitting.