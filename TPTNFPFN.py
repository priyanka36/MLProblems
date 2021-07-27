from matplotlib import pyplot as plt

def true_positive(y_true,y_pred):
    '''
    Function that calculates True Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true positives
    '''
    tp=0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1

    return tp 

def true_negative(y_true,y_pred):
    '''
    Function that calculates False Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true negatives
    '''
    tn=0
    for yt,yp in zip(y_true,y_pred):
        if yt == 0 and yp == 0:
            tn+=1
    return tn

def false_positive(y_true,y_pred):
    '''
    Function that calculates False Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false positives

    '''
    fp = 0
    for yt,yp in zip(y_true,y_pred):
        if yt == 0 and yp == 1:
            fp+=1
    return fp

def false_negative(y_true,y_pred):
    '''
    Function that calculates False Negatives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false negatives

    '''
    fp = 0
    for yt,yp in zip(y_true,y_pred):
        if yt == 1 and yp == 0:
            fp+=1
    return fp


l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]

tp=true_positive(l1, l2)
print(tp)

tn=true_negative(l1, l2)
print(tn)
    
fp=false_positive(l1, l2)
print(fp)

fn=false_negative(l1, l2)
print(fn)

def accuracy(y_true,y_pred):
    """
    Function to calculate accuracy using tp/tn/fp/fn
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    accuracy_score = (tp + tn) / (tp + tn + fp + fn)
    return accuracy_score


def precision(y_true,y_pred):
    """
    Function to calculate precision
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: precision score
    """
    tp = true_positive(y_true,y_pred)
    fp = false_positive(y_true,y_pred)
    precision = tp/(tp+fp)
    print(f"{precision} is the precision")
    return precision

precision(l1,l2)


def recall(y_true,y_pred):
    '''
    Function to calculate recall
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: precision score
    '''
    tp = true_positive(y_true,y_pred)
    fn = false_negative(y_true,y_pred)
    recall = tp/(tp+fn)
    print(f"{recall} is the recall")

    return recall

recall(l1, l2)


y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

y_pred = [0.02638412, 0.11114267, 0.31620708,
0.0490937, 0.0191491, 0.17554844,
0.15952202, 0.03819563, 0.11639273,
0.079377,
0.08584789, 0.39095342,
0.27259048, 0.03447096, 0.04644807,
0.03543574, 0.18521942, 0.05934905,
0.61977213, 0.33056815]

precisions = []
recalls = []
predicted=[]
thresholds = [0.0490937 , 0.05934905, 0.079377,
0.08584789, 0.11114267, 0.11639273,
0.15952202, 0.17554844, 0.18521942,
0.27259048, 0.31620708, 0.33056815,
0.39095342, 0.61977213]

for i in thresholds:
    temp_prediction=[1 if x>=i else 0 for x in y_pred]
    predicted.append(temp_prediction)
    p = precision(y_true,temp_prediction)
    r = recall(y_true,temp_prediction)
precisions.append(p)
recalls.append(r)
print(precisions)
print(recalls)

plt.figure(figsize=(7,7))
plt.plot(recalls,precisions)
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)