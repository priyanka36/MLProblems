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