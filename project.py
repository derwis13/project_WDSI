import os
import xml.etree.ElementTree as ET
import shutil

import numpy
from sklearn.cluster import KMeans
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def loadAndSplit(file_path,file_path_2):

    list_names = os.listdir(file_path)

    information=[]
    countOfSpeedlimit=0
    countOfOther=0
    allphoto=len(list_names)
    for each in list_names:
        inf = []
        tree=ET.ElementTree(file=file_path+each)
        root=tree.getroot()
        file_name=root[1].text
        for child in root.iter('object'):
            for bndbox in child.iter("bndbox"):
                if child[0].text=='speedlimit':
                    countOfSpeedlimit+=1
                else:
                    child[0].text='other'
                inf.append(child[0].text)
        information.append([file_name,each, inf,])
    countOfOther=allphoto-countOfSpeedlimit
    countOfOther_1=0
    countOfSpeedlimit_1=0

    for each in information:
        if each[2][0]=='speedlimit':
            if countOfSpeedlimit/3>countOfSpeedlimit_1:
                shutil.copy(file_path_2+each[0],'train/images')
                shutil.copy(file_path+each[1],'train/annotations')
                countOfSpeedlimit_1+=1
            if countOfSpeedlimit / 3 <= countOfSpeedlimit_1:
                shutil.copy(file_path_2 + each[0], 'test/images')
                shutil.copy(file_path + each[1], 'test/annotations')
        else:
            if countOfOther/3>countOfOther_1:
                shutil.copy(file_path_2+each[0],'train/images')
                shutil.copy(file_path+each[1],'train/annotations')
                countOfOther_1+=1
            if countOfOther/3<=countOfOther_1:
                shutil.copy(file_path_2 + each[0], 'test/images')
                shutil.copy(file_path + each[1], 'test/annotations')

def load(annotations_path,photo_path):
    list_names = os.listdir(annotations_path)

    data = []
    for each in list_names:
        tree = ET.ElementTree(file=annotations_path + each)
        root = tree.getroot()
        file_name = root[1].text
        for child in root.iter('object'):
            for bndbox in child.iter("bndbox"):
                class_id=0
                if child[0].text == 'speedlimit':
                    class_id='speedlimit'
                else:
                    class_id='other'
                img=cv2.imread(photo_path+file_name)[int(bndbox[1].text):int(bndbox[3].text),int(bndbox[0].text):int(bndbox[2].text)]
                data.append({'image':img,'label':class_id})
    return data

def boVW(data):
    sift = cv2.SIFT_create()
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)
    descriptor_list = []
    dictionary = {}
    for dict in data:
        descriptor_list=[]
        features = []
        kp, des = sift.detectAndCompute(dict['image'], None)
        if des is not None:
            descriptor_list.extend(des)
            bow.add(des)
    vocabulary=bow.cluster()
    return vocabulary
def extractingFeautres(vocabulary,data):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow_extr = cv2.BOWImgDescriptorExtractor(sift,flann)
    bow_extr.setVocabulary(vocabulary)
    data_feautres=[]
    category=[]
    for dict in data:
        histogram=[]
        key=sift.detect(dict['image'])
        if key is True:
            histogram=np.zeros(128)
        else:
            histogram=bow_extr.compute(dict['image'],key)

        #print(histogram[0])

        data_feautres.append({'data': histogram, 'label': dict['label']})
    return data_feautres

def train(data):
    X=[]
    y=[]
    for dict in data:
        try:
            X.append(dict['data'][0])
            y.append(dict['label'])
        except:
            X.append(np.zeros(128))
            y.append('other')
            #do nothing

    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X,y)
    return clf

def predict(clf,data_test):
    labels=[]
    X_data=[]
    for dict in data_test:
        #print(np.zeros(128))
        try:
            X_data.append(dict['data'][0])
            labels.append(dict['label'])
        except:
            X_data.append(np.zeros(128))
            labels.append('other')


    #print(X_data)
    pred=clf.predict(X_data)
    print(pred)
    print(accuracy_score(labels,pred))

def classification(vocabulary,clf):
    file_path='images/'

    command = input('command')
    p_files = input('p_files')
    bndbox = []
    data=[]
    if command=='classify':
        for p in range(int(p_files)):
            file_name = input('file_name')
            n_pictures = input('n_pictures')
            for n in range(int(n_pictures)):
                xmin, xmax, ymin, ymax = [int(x) for x in input('xmin...').split(' ')]
                img = cv2.imread(file_path+file_name)[int(ymin):int(ymax),int(xmin):int(xmax)]
                cv2.imshow('image',img)
                cv2.waitKey(0)
                data.append({'image': img, 'label': None})

    data=extractingFeautres(vocabulary,data)
    predict(clf,data)

    #print(clf.predict(data[0]['data'][0]))


data_train=load('train/annotations/','train/images/')
data_test=load('test/annotations/','test/images/')
vocabulary=boVW(data_train)
clf=train(extractingFeautres(vocabulary,data_train))
predict(clf,extractingFeautres(vocabulary,data_test))
#bndbox=[0,100,0,100]
#classification(vocabulary,clf)







