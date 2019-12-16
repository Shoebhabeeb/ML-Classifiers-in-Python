

def Break_interval_list(interval_list):
    def Break_intervals(interval):
        b = interval.split(',')
        c = (b[0][1:],b[1][:-1])
        return c
    return list(map(Break_intervals, interval_list))

def Probability_list(training_df):
    import pandas as pd
    import numpy as np
    dataFrame = training_df.copy()
    probability_dict = {}
    probability_dict['target'] = {}
    labels = dataFrame['target'].unique()
    features = dataFrame.columns[:-1]
    
    for feature in features:
        temp_data = pd.qcut(dataFrame[feature],10)
        binned_data = list(map(lambda x: str(x), temp_data))
        dataFrame[feature] = binned_data

    for label in labels:
        temp_df = dataFrame[dataFrame['target'] == label]

        temp_dict = {}

        for feature in features:
            total_count = len(temp_df)
            individual_prob = temp_df[feature].value_counts().map(lambda x: x/total_count)
            a = list(individual_prob.values)
            b = list(Break_interval_list(individual_prob.index))
            temp_dict[feature] = list(map(lambda x,y: (x[0],x[1],y), b,a))

        probability_dict[label] = temp_dict

        num_instances = len(dataFrame['target'])

        probability_dict['target'][label] = len(temp_df)/num_instances

    return probability_dict

def keywithmaxval(d):
    """ a) create a list of the dict's keys and values; 
    b) return the key with the max value
    Source: #https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    """  
    v = list(d.values())
    k=list(d.keys())
    return k[v.index(max(v))]

def Test_Probability(X, probability_list, features):
    import pandas as pd
    import numpy as np
    X = np.array(X)
    labels = list(probability_list['target'].keys())
    size = X.shape[0]
    df1 = pd.DataFrame(data = X.reshape((1,size)), columns = features)
    final_prob = {}
    for label in labels:
        accumulated_prob = 1
        for feature in features:
            prob_feat = probability_list[0][feature]
            test_point = float(df1[feature])
            try:
                ind_prob = list(map(lambda x: x[2], filter(lambda y: test_point > float(y[0]) and test_point <= float(y[1]), prob_feat)))[0]
            except:
                ind_prob = 0
            accumulated_prob *= ind_prob
        accumulated_prob *= probability_list['target'][label]
        final_prob[label] = accumulated_prob

    return keywithmaxval(final_prob)

def Naive_Bayes_Classifier(train_dataFrame, test_data):
    import numpy as np
    import pandas as pd
    features = train_dataFrame.columns[:-1]
    probability_list = Probability_list(train_dataFrame)
    return Test_Probability(test_data, probability_list, features)
    

def Naive_Bayes_Multiple(dataFrame, test_data):
    import numpy as np
    import pandas as pd
    prediction = []
    for i in range(len(test_data)):
        prediction.append(Naive_Bayes_Classifier(dataFrame, test_data[i]))
    
    return prediction