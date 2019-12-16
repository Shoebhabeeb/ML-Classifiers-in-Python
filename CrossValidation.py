import math
import pandas as pd
import numpy as np

def Train_Test_Split(dataFrame, k_fold = 10):
    
    n = math.ceil(len(dataFrame)/k_fold)
    
    train_test_list = []
        
    for i in range(k_fold):
        temp_tuple = (pd.concat([dataFrame[:(i*n)],dataFrame[((i+1)*n):]]),dataFrame[(i*n):((i+1)*n)])
        train_test_list.append(temp_tuple)
        
    return train_test_list
        
def Classifier(train_set,test_set, prediction):
    import pandas as pd
    import numpy as np
    X_train = train_set.iloc[:,:-1]
    y_train = train_set.iloc[:,-1]
    
    X_test = test_set.iloc[:,:-1]
    y_test = np.array(test_set.iloc[:,-1])  
    
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == prediction[i]:
            correct += 1
    return correct / float(len(y_test))

def KFold_CV(real,predicted,dataFrame):
    import pandas as pd
    import numpy as np
    total = len(dataFrame)
    correct = 0
    for i in range(len(real)):
        for j in range(len(real[i])):
            if (real[i][j] == predicted[i][j]):
                correct += 1
    return(correct*100/total)