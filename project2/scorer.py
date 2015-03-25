
def calculate_score(predicted_labels, actual_labels):
    '''
    Expects both inputs in format of list with elements as dicts as follows:
    {
        Y: predicted / actual label for Y
        Z: predicted / actual label for Z
    }
    '''
    yscore = 0.0
    zscore = 0.0
    n = len(predicted_labels)
    for i in range(n):
        if predicted_labels[i]['Y'] != actual_labels[i]['Y']:
            yscore += 1.0
        if predicted_labels[i]['Z'] != actual_labels[i]['Z']:
            zscore += 1.0
    return (yscore + zscore)/(2*n)

def get_grid_search_scorer(n):
    return lambda y, y_pred: 0.0 if y == y_pred else 1.0/n
