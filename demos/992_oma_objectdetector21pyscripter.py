from imageai.Detection import ObjectDetection
import wget
import sys
print("this first line fine")


def GraphViz(node):
    d = Graph(node)

    from graphviz import Digraph
    dot = Digraph("Graph", strict=False)
    dot.format = "png"

    def rec(nodes, parent):
        for d in nodes:
            if not isinstance(d, dict):
                dot.node(d, shape=d._graphvizshape)
                dot.edge(d, parent)
            else:
                for k in d:
                    dot.node(k._name, shape=k._graphvizshape)
                    rec(d[k], k)
                    dot.edge(k._name, parent._name)
    for k in d:
        dot.node(k._name, shape=k._graphvizshape)
        rec(d[k], k)
    return dot


detector = ObjectDetection()

url = "http://www.kleiner.ch/images/italo_max_train.jpg"
url="https://softwareschulecode.files.wordpress.com/2019/12/tee_film4.png?w=750"
destination = "./input/film_tee_train.jpg"

model_path = "./crypt/models/yolo-tiny.h5"
input_path = "./crypt/input/basel401.jpg"      #testcase3.png #twinwiz.jpg"

#wget.download(url, out=destination) #, useragent= "maXbox")
#input_path = destination
#output_path = "./crypt/output/manmachine.jpg"
output_path = sys.argv[1]

#using the pre-trained TinyYOLOv3 model,
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)

#loads model of path specified above using setModelPath() class method.
detector.loadModel()

custom=detector.CustomObjects(person=True,laptop=True,car=False,train=True,
                       clock=True, chair=False, bottle=False, keyboard=True)

detections=detector.detectCustomObjectsFromImage(custom_objects=custom,  \
                  input_image=input_path, output_image_path=output_path, \
                                     minimum_percentage_probability=40.0)

for eachItem in detections:
    print(eachItem["name"] ," : ", eachItem["percentage_probability"])

print("integrate image detector compute ends...")

import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")
figfile = StringIO()
plt.savefig(figfile, format="svg")
plt.show()
print(df.info())


#df = df.dropna()

"""
df['sex'] = df['sex'].fillna('MALE')
df.drop(df[df['sex']=='.'].index, inplace=True)

df = df.copy()
target = 'sex'
encode = ['species','island']

print(df.describe())

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'MALE':1, 'FEMALE':2}
def target_encode(val):
    try:
      return target_mapper[val]
    except:
      pass

df['sex'] = df['sex'].apply(target_encode)

print(df.isnull().sum())

#my imputer
#df[df==np.inf]=np.nan
#df.fillna(df.mean(), inplace=True)

#df = df.dropna()
"""

#X = df.drop("species", axis=1)

print(df.isnull().sum())
df = df.dropna()
df = df.drop(df[df['species'] == 'Gentoo'].index)

encode = ['island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
print(dummy)

#X = df[["bill_length_mm", "bill_depth_mm","flipper_length_mm","island_Dream"]]
X = df[["bill_length_mm", "bill_depth_mm","flipper_length_mm"]]
#X = df["culmen_length_mm", "culmen_depth_mm"]
#df['species'] = df[df['species']['Chinstrap','Adelie']]
#df["species"] = df[df['species'] == 'Gentoo','Adelie']
y = df["species"]

#df['sex'] = df['sex'].fillna('MALE')
#X=X[:, ~np.isnan(X).any(axis=1)]
#X=X[:, ~np.isnan(X).any(axis=1)]
#y=y[~y.isnull()]

print(X.shape)
print(y.shape)

from sklearn import preprocessing
X = preprocessing.scale(X)

#splitting the data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=13)

# model fitting and prediction

from sklearn.linear_model import LogisticRegression

#new_df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns, index=df.index)

#X_train = X_train.isnan()
#import numpy as np
#X_train=X_train[:, ~np.isnan(X_train).any(axis=0)]
#y_train=y_train[~y_train.isnull()]
#pd.options.mode.use_inf_as_na = True

"""
X_train[:, ~np.isnan(X_train).any(axis=0),]
X_train[:, ~np.isfinite(X_train).any(axis=0)]
y_train[~np.isnan(y_train)]
y_train[~np.isfinite(y_train)]
"""
print(X_train.shape)
print(y_train.shape)

model = LogisticRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print(pred)

# checking performance of model

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print('CONFUSION MATRIX')
print(confusion_matrix(y_test, pred))
print('CLASSIFICATION REPORT\n')
print(classification_report(y_test, pred))

sns.heatmap(confusion_matrix(y_test, pred), annot=True)
plt.show()

# ROC CURVE

from collections import Counter
print(Counter(y_train)) # y_true must be your labels


print('ROC CURVE')
train_probs = model.predict_proba(X_train)
train_probs1 = train_probs[:, 0]
fpr0, tpr0, thresholds0 = roc_curve(y_train, train_probs1, pos_label='Adelie')

test_probs = model.predict_proba(X_test)
test_probs1 = test_probs[:, 0]
fpr1, tpr1, thresholds1 = roc_curve(y_test, test_probs1, pos_label='Adelie')

plt.plot(fpr0, tpr0, marker='.', label='train')
plt.plot(fpr1, tpr1, marker='.', label='validation')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

from sklearn.cluster import KMeans
print("CLUSTERING ON CULMEN LENGTH AND CULMEN DEPTH")
X = df[["bill_length_mm","bill_depth_mm"]]
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X.loc[:,"bill_length_mm"], X.loc[:,"bill_depth_mm"],c=y_kmeans,s=50,cmap="viridis")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
plt.show()




