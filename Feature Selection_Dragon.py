# -*- coding: utf-8 -*-
"""
Created on Fri May  6 23:02:28 2022

@author: ahmad
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:55:02 2022

@author: ahmad
"""

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#from  niapy  import nd.array
#import pyswarms as ps 
#تحميل البيانات
x = load_breast_cancer().data
y = load_breast_cancer().target
# تقسيم البيانات
X_tr, X_test, y_tr, y_test = train_test_split(x, y, test_size=0.3, random_state=1, shuffle =True)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=1, shuffle =True)

#c1 = LogisticRegression(multi_class='multinomial', random_state=1)
c1 = svm.SVC(kernel='linear')
c2 = KNeighborsClassifier(n_neighbors=1)
c3 = GaussianNB()
c4 = DecisionTreeClassifier(criterion = 'gini', max_depth=4, random_state=1)
c5 = RandomForestClassifier(n_estimators=120,criterion='gini',max_depth=5,random_state=1)


hard_vote = VotingClassifier(estimators=[ ('SVM', c1),('KN', c2),('GNB',c3),('DT',c4),('RF',c5)], voting='hard')
#ec1 = 
hard_vote.fit(X_train, y_train)
y_pred = hard_vote.predict(X_test)
#hard_vote.accuracy score(y_test,y_pred)
#y_pred = hard_vote.predict(x_test)
#print(hard_vote.score(X_test,y_test)*100)
from sklearn import metrics
print(" Befor Feature Selection")
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred)*100)
print("Precision:",metrics.precision_score(y_test, y_pred)*100)
print("Recall:   ",metrics.recall_score(y_test, y_pred)*100)
print("f1_score: ",metrics.f1_score(y_test, y_pred,average='binary')*100)
#from sklearn.datasets import load_breast_cancer



####---------- Feature Slection ------------------------
## Maher 

import pandas as pd
data = load_breast_cancer()
X2_train=pd.DataFrame(X_train,columns=data['feature_names'])
X2_val=pd.DataFrame(X_val,columns=data['feature_names'])
X2_test=pd.DataFrame(X_test,columns=data['feature_names'])



from zoofs import DragonFlyOptimization

def objective_function_topass(hard_vote,X_train, y_train, X_test, y_test):      
   hard_vote.fit(X_train,y_train)  
   
   P=metrics.accuracy_score(y_test,hard_vote.predict(X_test))
   return P

algo_object=DragonFlyOptimization(objective_function_topass,n_iteration=4,
                                    population_size=20,method='sinusoidal',minimize=True)


from sklearn.ensemble import VotingClassifier
hard_vote = VotingClassifier(estimators=[ ('SVM', c1),('KN', c2),('GNB',c3),('DT',c4),('RF',c5)], voting='hard')
#lgb_model = lgb.LGBMClassifier()                                       
# fit the algorithm
algo_object.fit(hard_vote,X2_train, y_train, X2_val, y_val,verbose=True)
algo_object.best_feature_list
print(algo_object.best_feature_list)
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(algo_object.best_feature_list)))


#selected_feat=['mean perimeter', 'mean concavity', 'mean concave points', 'perimeter error', 'fractal dimension error']
selected_feat = algo_object.best_feature_list

X_train_sele_fea = pd.DataFrame(X2_train,columns=selected_feat)
X_test_sele_fea = pd.DataFrame(X2_test,columns=selected_feat)

X_train_sele_fea2=X_train_sele_fea.to_numpy()
X_test_sele_fea2=X_test_sele_fea.to_numpy()

hard_vote.fit(X_train_sele_fea2, y_train)

y_pred = hard_vote.predict(X_test_sele_fea2)
#hard_vote.accuracy score(y_test,y_pred)
#y_pred = hard_vote.predict(x_test)
#print(hard_vote.score(X_test,y_test)*100)
from sklearn import metrics
print(" After Feature Selection")
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred)*100)
print("Precision:",metrics.precision_score(y_test, y_pred)*100)
print("Recall:   ",metrics.recall_score(y_test, y_pred)*100)
print("f1_score: ",metrics.f1_score(y_test, y_pred,average='binary')*100)





#selected_feat.to_csv('selected_feat.csv')

# Export Selected Columns to CSV File
#df.to_csv('my_file.csv',index=False, columns=['algo_object.best_feature_list'])          

#print('Subset accuracy:', hard_vote .score(X_train[:, selected_feat], y_test))

#model_all.fit(X_train, y_train)
#print('All Features Accuracy:', hard_vote.score(X_test, y_test))
