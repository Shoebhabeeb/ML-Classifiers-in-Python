import pandas as pd
import numpy as np

#KNN Classifier, contains Minkosky Distance function

def KNN_Classifier(dataFrame, unknown, K, min = 0, max = 1, p=2):
    import pandas as pd
    import numpy as np
    def Minkosky_Distance(dataFrame, unknown, p, k, min_normalization = 0, max_normalization = 1):
        features = dataFrame.columns[:-1]
        unknown_df = pd.DataFrame(unknown.reshape(1,len(features)), columns = features)
        df2 = pd.DataFrame()
        
        for feature in features:
            attr_max = dataFrame.describe()[feature][-1]
            attr_min = dataFrame.describe()[feature][3]
            df2[feature] = dataFrame[feature].map(lambda x: ((x-attr_min)*(max_normalization-min_normalization)/(attr_max-attr_min))-min_normalization)
            z = unknown_df[feature]
            unknown_df[feature] = ((z-attr_min)*(max_normalization-min_normalization)/(attr_max-attr_min))-min_normalization
        df2['target'] = dataFrame['target']
        X = df2.iloc[:,:-1]
        data = map(lambda row: list(map(lambda x,y: (abs(x-y))**p, row, unknown)),X.values)

        new_df = pd.DataFrame(data, columns = features)
        result = new_df.sum(axis=1).map(lambda total: total**(1/p)).sort_values(ascending=True)[:K]
        return result
    
    
    result = Minkosky_Distance(dataFrame, unknown, p, K, min_normalization = min, max_normalization = max)
    
    return dataFrame.iloc[list(result.index),:]['target'].value_counts().index[0]
    

def KNN_Multiple(dataFrame, test_data, K=3, min = 0, max = 1, p=2):
    import pandas as pd
    import numpy as np
    prediction = []
    for i in range(len(test_data)):
        prediction.append(KNN_Classifier(dataFrame, np.array(test_data.iloc[i]), K=3, min = 0, max = 1, p=2))
    
    return prediction