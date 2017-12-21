from PIL import Image
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from feature import NPDFeature
from ensemble import AdaBoostClassifier

# step 1: read datasets
def convert_images(path,label):
    imgs = os.listdir(path)
    X = []
    y = []
    for img in imgs:
        img_path = path+"/"+img
        im = Image.open(img_path).convert("L").resize((24,24))
        X.append(np.array(NPDFeature(np.asarray(im)).extract()))
        y.append(label)
    X = np.array(X).reshape(len(imgs),-1)
    y = np.array(y).reshape(len(imgs),1)
    return X,y

# step 2: process datasets, extract NPD features
def process_images():
    a,b = convert_images("datasets/original/face",1)
    c,d = convert_images("datasets/original/nonface",-1)
    X = np.vstack((a,c))
    y = np.vstack((b,d))
    with open("datasets/processed/feature",'wb') as feature:
        pickle.dump(X,feature)
    with open("datasets/processed/label",'wb') as label:
        pickle.dump(y,label)


if __name__ == "__main__":
    print("Processing Images-------")
    # process_images()
    print("Loading data------------")
    with open("datasets/processed/feature",'rb') as feature:
        X = pickle.load(feature)
    with open("datasets/processed/label",'rb') as label:
        y = pickle.load(label)
    print("Shape of data after processing: ",X.shape, y.shape)

    # spliting dataset
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size = 0.30, random_state=42)
    print(X_train.shape, y_train.shape,X_valid.shape,y_valid.shape)

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),10)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_valid)
    acc = np.mean(y_pred == y_valid.reshape(-1,))
    print(acc)
    with open("datasets/report.txt",'wb') as file:
        report = classification_report(y_valid, y_pred, target_names = ['face', 'nonface'])
        file.write(report.encode())