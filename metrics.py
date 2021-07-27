l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]


def true_positive(y_true,y_pred):
    '''
    Function to calculate True Positives 
    :param y_true:list of true values
    :param y_pred:list of predicted values
    :return :number of true positives

    tp = 0
    '''
    for yt,yp in zip(y_true,y_pred):
        if yt == 1 and yp == 1:
            tp+=1
    return tp

def true_negative(y_true,y_pred):
    '''
    Function to calculate True Positives 
    :param y_true:list of true values
    :param y_pred:list of predicted values
    :return :number of true positives

    tp = 0
    '''
    for yt,yp in zip(y_true,y_pred):
        if yt == 0 and yp == 0:
            fn+=1
    return fn

def false_positive(y_true,y_pred):
    '''
    Function to calculate True Positives 
    :param y_true:list of true values
    :param y_pred:list of predicted values
    :return :number of true positives

    tp = 0
    '''
    for yt,yp in zip(y_true,y_pred):
        if yt == 1 and yp == 0:
            tp+=1
    return tp

def false_negative(y_true,y_pred):
    '''
    Function to calculate True Positives 
    :param y_true:list of true values
    :param y_pred:list of predicted values
    :return :number of true positives

    tp = 0
    '''
    for yt,yp in zip(y_true,y_pred):
        if yt == 0 and yp == 1:
            tp+=1
    return tp


true_positive(l1, l2)
false_positive(l1, l2)
false_negative(l1, l2)
true_negative(l1, l2)