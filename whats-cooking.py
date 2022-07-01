import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm

lgr = LogisticRegression(random_state=0)
rf=RandomForestClassifier(random_state=1)
nb=MultinomialNB()
dt=DecisionTreeClassifier(random_state=0)
sc=svm.SVC()
grb = GradientBoostingClassifier(n_estimators=10)

d_c = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian',
 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole',
 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']

data = pd.read_json('train.json')


print(data['cuisine'].unique())
x=data['ingredients']
y=data['cuisine'].apply(d_c.index)
data['all_ingredients']=data['ingredients'].map(';'.join)

cv=CountVectorizer()
x= cv.fit_transform(data['all_ingredients'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
#
# lgr.fit(x_train,y_train)
# rf.fit(x_train, y_train)
# nb.fit(x_train, y_train)
# dt.fit(x_train, y_train)
# sc.fit(x_train, y_train)
grb.fit(x_train,y_train)

# lgr_prdt= lgr.predict(x_test)
# rf_prdt= rf.predict(x_test)
# nb_prdt = nb.predict(x_test)
# dt_prdt = dt.predict(x_test)
# sc_prdt = sc.predict(x_test)
grb_prdt =grb.predict(x_test)

# print('Logistic_regreassion',accuracy_score(y_test,lgr_prdt))
# print('RandomForest', accuracy_score(y_test, rf_prdt))
# print('NaiveBayes', accuracy_score(y_test, nb_prdt))
# print('DecisionTreeClassifier', accuracy_score(y_test, dt_prdt))
# print('svm', accuracy_score(y_test, sc_prdt))
print('grb',accuracy_score(y_test,grb_prdt))

'''
Accuracy Score:
Logistic_regreassion 0.7841609050911377
RandomForest 0.7546197360150848
NaiveBayes 0.7208045254556883
DecisionTreeClassifier 0.6369578881206788
supportvectored  0.8049025769956003
GradientBoostingClassifier 0.6281583909490887
'''
