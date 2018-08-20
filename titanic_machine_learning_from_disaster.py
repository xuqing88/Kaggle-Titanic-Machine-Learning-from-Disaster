import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Macro Definition
train_data_ratio = 0.8
test_data_ratio = 0.2
cv_data_ratio = 0
Train_flag = 'Training data'
Test_flag = 'Testing data'
Cross_validataion_flag = 'Cross-Val data'

sibsp_scaler_file_name = "sibsp_scaler.save"
parch_scaler_file_name = "parch_scaler.save"
fare_scaler_file_name = "fare_scaler.save"
age_scaler_file_name = "age_scaler.save"

# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked {'SibSp','Parch',}
feature_selection = ['Pclass','Sex','Age','Fare','Embarked']

def survived_process(survived):
     return survived.values


def pclass_process(pclass):
    result  = pd.get_dummies(pclass)
    return result.values


def sex_process(sex):
    result = pd.get_dummies(sex)
    return result.values


def age_process(age):
    """
    Separate the age into five different catalogs:
    Children: 0~14
    Youth: 15~24
    Adults: 25~64
    Seniors: 65+
    Unknown: No information
    """
    # Method 1 : Categorical
    max_age = age.max(skipna=True)
    cat_type = {'Children':[1,0,0,0,0],'Youth':[0,1,0,0,0],'Adults':[0,0,1,0,0],'Senirors':[0,0,0,1,0],'Others':[0,0,0,0,1]}
    age_data_size = age.shape[0]
    result = []
    for i in range(age_data_size):
        if age.values[i] <= 15:
            if i == 0:
                result = np.append(result,cat_type['Children']).reshape(1,-1)
            else:
                result = np.vstack((result,cat_type['Children']))
        elif age.values[i] > 15 and age.values[i]<=25:
            if i == 0:
                result = np.append(result,cat_type['Youth']).reshape(1,-1)
            else:
                result = np.vstack((result,cat_type['Youth']))
        elif age.values[i] > 25 and age.values[i] <= 65:
            if i == 0:
                result = np.append(result,cat_type['Adults']).reshape(1,-1)
            else:
                result = np.vstack((result, cat_type['Adults']))
        elif age.values[i] > 65 and age.values[i] <= max_age:
            if i == 0:
                result = np.append(result,cat_type['Senirors']).reshape(1,-1)
            else:
                result = np.vstack((result, cat_type['Senirors']))
        else:
            if i == 0:
                result = np.append(result,cat_type['Others']).reshape(1,-1)
            else:
                result= np.vstack((result,cat_type['Others']))
    return result

    #Method 2: Normalization
    # tmp = age.values.reshape(-1, 1)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # age_normalization = min_max_scaler.fit(tmp)
    # joblib.dump(age_normalization,age_scaler_file_name)
    # return age_normalization.transform(tmp)


def sibsp_process(sibsp):
    """
    Value =   0   1  2  3  4  5 8
    Count = 608 209 28 16 18 5 7
    Try just normalization first
    """
    tmp = sibsp.values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    sibsp_normalization = min_max_scaler.fit(tmp)
    joblib.dump(sibsp_normalization,sibsp_scaler_file_name)
    return sibsp_normalization.transform(tmp)

def parch_process(parch):
    # same process as sibsp
    """
    Value =   0   1  2  3  4  5 6
    Count = 678 118 80  5  4  5 1
    """
    tmp = parch.values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    parch_normalization = min_max_scaler.fit(tmp)
    joblib.dump(parch_normalization,parch_scaler_file_name)
    return parch_normalization.transform(tmp)


def fare_process(fare):
    #Method 1:just normalize it
    tmp = fare.values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    fare_normalization = min_max_scaler.fit(tmp)
    joblib.dump(fare_normalization,fare_scaler_file_name)
    return fare_normalization.transform(tmp)

    #Method 2: Categorical:
    # """
    # Fare 0-10: Cat1
    # Fare 10-50:Cat2
    # Fare 50-100: Cat3
    # Fare 100-200:Cat4
    # Fare 200-300: Cat5
    # Fare >300: Cat6
    # """
    # max_fare = fare.max(skipna=True)
    # cat_type = {'CAT1': [1, 0, 0, 0], 'CAT2': [0, 1, 0, 0], 'CAT3': [0, 0, 1, 0],
    #             'CAT4': [0, 0, 0, 1]}
    # fare_data_size = fare.shape[0]
    # result = []
    # for i in range(fare_data_size):
    #     if fare.values[i] <= 7.91:
    #         if i == 0:
    #             result = np.append(result, cat_type['CAT1']).reshape(1, -1)
    #         else:
    #             result = np.vstack((result, cat_type['CAT1']))
    #     elif fare.values[i] > 7.91 and fare.values[i] <= 14.454:
    #         if i == 0:
    #             result = np.append(result, cat_type['CAT2']).reshape(1, -1)
    #         else:
    #             result = np.vstack((result, cat_type['CAT2']))
    #     elif fare.values[i] > 14.454 and fare.values[i] <= 31:
    #         if i == 0:
    #             result = np.append(result, cat_type['CAT3']).reshape(1, -1)
    #         else:
    #             result = np.vstack((result, cat_type['CAT3']))
    #     elif fare.values[i] > 31:
    #         if i == 0:
    #             result = np.append(result, cat_type['CAT4']).reshape(1, -1)
    #         else:
    #             result = np.vstack((result, cat_type['CAT4']))
    # return result

def family_size_process(family_size):
    size = family_size.shape[0]
    result =[]
    for i in range(size):
        if family_size[i]!=0:
            result = np.append(result,1)
        else:
            result =np.append(result,0)
    return result

def embarked_process(embarked):
    result = pd.get_dummies(embarked)
    return result.values

def data_preprocess(data, type):
    X = []
    Y = []
    for data_column in data.columns:
        if data_column == 'Survived':
           # print('process the survived column')
            Y = survived_process(data[data_column])

        if data_column == 'Pclass' and data_column in feature_selection:
            # print('Process the Pclass column')
            tmp = pclass_process(data[data_column])
            X = np.append(X, tmp).reshape(tmp.shape)
           # print('Add Pclass {}'.format(X.shape))

        # if data_column == 'Name':
          #  print('Warning: NOT Process the Name column !!!')

        if data_column == 'Sex'and data_column in feature_selection:
          #  print('Process the Sex column')
            tmp = sex_process(data[data_column])
            X = np.hstack((X, tmp))
         #   print('Add Sex {}'.format(X.shape))

        if data_column == 'Age'and data_column in feature_selection:
           #Method 1: Categorical
           tmp = age_process(data[data_column])
           X = np.hstack((X, tmp))

           #Method 2: Normalizaiton
          # age_filled  = data[data_column].fillna(value=0)
          # if type == Train_flag:
          #    #  print('Process the Age column')
          #   tmp = age_process(age_filled)
          #   X =np.hstack((X, tmp))
          #   #  print('Add Age {}'.format(X.shape))
          # elif type ==Test_flag or type==Cross_validataion_flag:
          #   age_normalizaiton = joblib.load(age_scaler_file_name)
          #   tmp = age_normalizaiton.transform(age_filled.values.reshape(-1,1))
          #   X= np.hstack((X,tmp))

        if data_column == 'SibSp'and data_column in feature_selection:
            if type == Train_flag:
           #     print('Process the Training SibSp column')
                tmp = sibsp_process(data[data_column])
                X = np.hstack((X, tmp))
          #      print('Add Sibso {}'.format(X.shape))
            elif type == Test_flag or type == Cross_validataion_flag:
           #     print('Process the Testing SibSp column')
                sibsp_normalization = joblib.load(sibsp_scaler_file_name)
                tmp = sibsp_normalization.transform(data[data_column].values.reshape(-1,1))
                X = np.hstack((X, tmp))
           #     print('Add Sibso {}'.format(X.shape))

        if data_column == 'Parch'and data_column in feature_selection:
            if type == Train_flag:
          #      print('Process the Parch column')
                tmp = parch_process(data[data_column])
                X = np.hstack((X,tmp))
         #       print('Add Patch {}'.format(X.shape))
            elif type == Test_flag or type == Cross_validataion_flag:
         #      print('Process the Testing Parch column')
                parch_normalization = joblib.load(parch_scaler_file_name)
                tmp = parch_normalization.transform(data[data_column].values.reshape(-1,1))
                X = np.hstack((X,tmp))
          #      print('Add Patch {}'.format(X.shape))

      #  if data_column == 'Ticket':
           # print('Waring: NOT Process the Ticket column !!')

        if data_column == 'Fare'and data_column in feature_selection:
            fare_filled = data[data_column].fillna(value=0)
            if type == Train_flag:
           #     print('Process the Fare column')
                tmp = fare_process(fare_filled)
                X= np.hstack((X,tmp))
           #     print('Add Fare {}'.format(X.shape))
            elif type == Test_flag or type == Cross_validataion_flag:
                #Method 2
                # tmp = fare_process(data[data_column])
                # X = np.hstack((X, tmp))
                # Method 1
                fare_normalization = joblib.load(fare_scaler_file_name)
                tmp = fare_normalization.transform(fare_filled.values.reshape(-1,1))
                X = np.hstack((X,tmp))
                #print('Add Fare {}'.format(X.shape))

      #  if data_column == 'Cabin':
           # print('Warning: NOT Process the Cabin column')

        if data_column == 'Embarked'and data_column in feature_selection:
           # print('Process the Embarked column')
            tmp = embarked_process(data[data_column])
            X= np.hstack((X,tmp))
           # print('Add Embarked {}'.format(X.shape))

    #Create New Features
    family_size = data['SibSp'].values + data['Parch'].values
    result = family_size_process(family_size).reshape(-1,1)
    X = np.hstack((X,result))

    return [X, Y]


# Part 1:  Data Engineering
# Load data from the train.csv and pre-process all the columns data, return X and Y
data_original = pd.read_csv('train.csv')
data_submit = pd.read_csv('test.csv')
# fig = plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.plot(data_original['PassengerId'],data_original['Fare'],'bo')
# fig.show()


# print(data_original.dtypes)
# Split the original data into three data set: Train-data_train, Test-data_test, Cross Valiadation- data_cv
#data_for_train,data_cv = train_test_split(data_original,test_size=cv_data_ratio)
data_train, data_test = train_test_split(data_original, test_size=0)
[X_train, Y_train]= data_preprocess(data_train, Train_flag)
#[X_test, Y_test] = data_preprocess(data_test, Test_flag)
[X_submit,Y_empty] = data_preprocess(data_submit, Test_flag)
#[X_cv,Y_cv] = data_preprocess(data_cv,Cross_validataion_flag)


# Part2 : Model Selection
# Random forest

# RF-Grid Search
# print('Training the data with the Random forest.........')
# num_estimators = [10,20,30]
# max_features = ['auto','log2']
# para_grids ={'n_estimators':num_estimators,'max_features':max_features}
# grid_search= GridSearchCV(estimator=RandomForestClassifier(),param_grid=para_grids)
# grid_search.fit(X_train,Y_train)
# print('Random Forest Best Score:{} with{}'.format(grid_search.best_score_, grid_search.best_params_))
# means = grid_search.cv_results_['mean_test_score']
# stds = grid_search.cv_results_['std_test_score']
# params = grid_search.cv_results_['params']
# for mean, stdev,param in zip(means,stds,params):
#     print('Random Forest Score = {}, Std = {}, with{}'.format(mean, stdev,param))
# test_score = grid_search.best_estimator_.score(X_test,Y_test)
# print("RF Score:  Training Accuracy = {}, Test Accuracy = {}".format(grid_search.best_score_, test_score))

# RF - Manual Search
# train_score_result = []
# test_score_result = []
# for i in np.arange(start=1,stop=15,step=1):
#     rfclf = RandomForestClassifier(n_estimators=14,max_features=i,max_depth=6)
#     rfclf.fit(X_train,Y_train)
#     train_score = rfclf.score(X_train,Y_train)
#     test_score = rfclf.score(X_test,Y_test)
#     #print("Random Forest Score:  Training Accuracy = {}, Test Accuracy = {}".format(train_score, test_score))
#     train_score_result = np.append(train_score_result,train_score)
#     test_score_result=np.append(test_score_result,test_score)
# fig = plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.plot(np.arange(start=1,stop=15,step=1).reshape(-1,1), train_score_result,linestyle = '--',color = 'green',label = 'Training')
# ax.plot(np.arange(start=1,stop=15,step=1).reshape(-1,1),test_score_result, linestyle = ':', color = 'yellow',label = 'Test')
# fig.show()

# RF - Final Training
rfclf = RandomForestClassifier(n_estimators=14, max_features=12,max_depth=6)
rfclf.fit(X_train,Y_train)
print(rfclf.feature_importances_)
train_score = rfclf.score(X_train,Y_train)
#test_score = rfclf.score(X_test,Y_test)
print("Random Forest Score:  Training Accuracy = {}".format(train_score))
Y_submit = rfclf.predict(X_submit)
prediction = pd.DataFrame(Y_submit,columns=['Survived']).to_csv('prediction.csv')

# Decision Tree
# dtclf = DecisionTreeClassifier()
# dtclf.fit(X_train,Y_train)
# train_score = dtclf.score(X_train,Y_train)
# test_score = dtclf.score(X_test,Y_test)
# print("Decision Tree Score:  Training Accuracy = {}, Test Accuracy = {}".format(train_score, test_score))


# SVM

#  SVM Grid Search
# print("Training the data with SVM model..........")
# Cs = [0.001]
# kernels = ['rbf','sigmoid']
# gammas = [0.0001,0.001,0.01,0.1]
# para_grids = {'C':Cs,'kernel':kernels,'gamma':gammas}
# grid_search = GridSearchCV(estimator=SVC(),param_grid=para_grids)
# grid_search.fit(X_train,Y_train)
# print('SVM Best Score:{} with{}'.format(grid_search.best_score_, grid_search.best_params_))
# means = grid_search.cv_results_['mean_test_score']
# stds = grid_search.cv_results_['std_test_score']
# params = grid_search.cv_results_['params']
# for mean, stdev,param in zip(means,stds,params):
#     print('SVM Score = {}, Std = {}, with{}'.format(mean, stdev,param))
#
# test_score = grid_search.best_estimator_.score(X_test,Y_test)
# print("SVM Score:  Training Accuracy = {}, Test Accuracy = {}".format(grid_search.best_score_, test_score))
# svmclf = SVC()
# svmclf.fit(X_train,Y_train)
# train_score = svmclf.score(X_train,Y_train)
# test_score = svmclf.score(X_test,Y_test)
#print("SVM Score:  Training Accuracy = {}, Test Accuracy = {}".format(train_score, test_score))

# SVM - Manual Search
# train_score_result = []
# test_score_result = []
# gamma_range = [0.02,0.04,0.06,0.08, 0.1, 0.12,0.14]
# #for i in gamma_range:
# for i in np.arange(start=2,stop=4,step=0.5):
#     svmclf = SVC(C=i, kernel='rbf', gamma=0.06)
#     svmclf.fit(X_train,Y_train)
#     train_score = svmclf.score(X_train, Y_train)
#     test_score = svmclf.score(X_test, Y_test)
#     #print("SVM Score:  Training Accuracy = {}, Test Accuracy = {}".format(train_score, test_score))
#     train_score_result = np.append(train_score_result, train_score)
#     test_score_result=np.append(test_score_result, test_score)
# fig = plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.plot(np.arange(start=2, stop=4, step=0.5).reshape(-1,1), train_score_result,linestyle = '--',color = 'green',label = 'Training')
# ax.plot(np.arange(start=2, stop=4, step=0.5).reshape(-1,1),test_score_result, linestyle = ':', color = 'yellow',label = 'Test')
# # ax.plot(gamma_range, train_score_result,linestyle = '--',color = 'green',label = 'Training')
# # ax.plot(gamma_range,test_score_result, linestyle = ':', color = 'yellow',label = 'Test')
# fig.show()


#SVM - Final Training
# svmclf = SVC(C=2.5, kernel='rbf',gamma = 0.06)
# svmclf.fit(X_train,Y_train)
# #print(svmclf.feature_importances_)
# train_score = svmclf.score(X_train,Y_train)
# #test_score = rfclf.score(X_test,Y_test)
# print("Random Forest Score:  Training Accuracy = {}".format(train_score))
# Y_submit = svmclf.predict(X_submit)
# prediction = pd.DataFrame(Y_submit, columns=['Survived']).to_csv('SVM_prediction.csv')



# Neural Network
# print("Training the data with Neural Network Model.........")
# hidden_layer_size =[(25)]
# activations = ['relu']
# learning_rate_inits = [0.01,0.05]
# max_iters = [600]
# learning_rates = ['constant','invscaling','adaptive']
# para_grids = {'hidden_layer_sizes':hidden_layer_size,'activation':activations,'learning_rate_init':learning_rate_inits, 'max_iter':max_iters,'learning_rate':learning_rates}
# grid_search = GridSearchCV(estimator=MLPClassifier(),param_grid=para_grids)
# grid_search.fit(X_train,Y_train)
# print('NN Best Score:{} with{}'.format(grid_search.best_score_, grid_search.best_params_))
# means = grid_search.cv_results_['mean_test_score']
# stds = grid_search.cv_results_['std_test_score']
# params = grid_search.cv_results_['params']
# for mean, stdev,param in zip(means,stds,params):
#     print('NN Score = {}, Std = {}, with{}'.format(mean, stdev,param))
#
# test_score = grid_search.best_estimator_.score(X_test,Y_test)
# print("NN Score:  Training Accuracy = {}, Test Accuracy = {}".format(grid_search.best_score_, test_score))

input('Press Enter to exit')