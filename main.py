import math
import os
import random
import xml.etree.ElementTree as ET
import shutil
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# TODO Jakość kodu i raport (4/4)


# TODO Skuteczność klasyfikacji 0.962 (4/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0


# TODO Skuteczność detekcji (/2)


def loadAndSplit(file_path,file_path_2):  #file_path: path to annotations files, file_path2: path to images files

    list_names = os.listdir(file_path)

    information=[]
    countOfSpeedlimit=0
    allphoto=len(list_names)
    for each in list_names:
        inf = []
        tree=ET.ElementTree(file=file_path+each)
        root=tree.getroot()
        file_name=root[1].text
        for child in root.iter('object'):
            if child[0].text=='speedlimit':
                countOfSpeedlimit+=1
            else:
                child[0].text='other'
            inf.append(child[0].text)
        information.append([file_name,each, inf])
    countOfOther=allphoto-countOfSpeedlimit
    countOfOther_1=0
    countOfSpeedlimit_1=0

    for each in information:
        if each[2][0]=='speedlimit':
            if countOfSpeedlimit/3>countOfSpeedlimit_1:
                shutil.copy(file_path_2+each[0],'/train/images')
                shutil.copy(file_path+each[1],'/train/annotations')
                countOfSpeedlimit_1+=1
            if countOfSpeedlimit / 3 <= countOfSpeedlimit_1:
                shutil.copy(file_path_2 + each[0], '/test/images')
                shutil.copy(file_path + each[1], '/test/annotations')
        else:
            if countOfOther/3>countOfOther_1:
                shutil.copy(file_path_2+each[0],'/train/images')
                shutil.copy(file_path+each[1],'/train/annotations')
                countOfOther_1+=1
            if countOfOther/3<=countOfOther_1:
                shutil.copy(file_path_2 + each[0], '/test/images')
                shutil.copy(file_path + each[1], '/test/annotations')

def load_ClassIdAndCropPhoto(annotations_path,photo_path):
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
                    if int(bndbox[0].text)>1 and int(bndbox[1].text)>1:
                        for i in range(5):
                            xmin = random.randrange(0, int(bndbox[0].text)-1, 1)
                            xmax = random.randrange(xmin+1, int(bndbox[0].text), 1)
                            ymin = random.randrange(0, int(bndbox[1].text)-1, 1)
                            ymax = random.randrange(ymin+1, int(bndbox[1].text), 1)
                            img=cv2.imread(photo_path+file_name)[ymin:ymax,xmin:xmax]
                            if xmax-xmin>1 and ymax-ymin>1:
                                data.append({'image':img,'label':'other'})
                else:
                    class_id='other'
                img=cv2.imread(photo_path+file_name)[int(bndbox[1].text):int(bndbox[3].text),int(bndbox[0].text):int(bndbox[2].text)]
                data.append({'image':img,'label':class_id})
    return data


def boVW(data):
    sift = cv2.SIFT_create()
    dict_size = 500#128
    bow = cv2.BOWKMeansTrainer(dict_size)
    for dict in data:
        kp, des = sift.detectAndCompute(dict['image'], None)
        if des is not None:
            bow.add(des)
    vocabulary=bow.cluster()
    return vocabulary

def extractingFeautres(vocabulary,data):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow_extr = cv2.BOWImgDescriptorExtractor(sift,flann)
    bow_extr.setVocabulary(vocabulary)

    data_feautres=[]

    for dict in data:
        histogram=[]
        key,des=sift.detectAndCompute(dict['image'],None)
        if des is not None:
            histogram=bow_extr.compute(dict['image'],key)
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
            # TODO Lepiej w ogole pominac takie przypadki.
            X.append(np.zeros(500))
            y.append('other')

    clf=RandomForestClassifier(n_estimators=1000)
    clf.fit(X,y)
    return clf

def predict(clf,data_test):
    labels=[]
    X_data=[]
    for dict in data_test:
        try:
            X_data.append(dict['data'][0])
            labels.append(dict['label'])
        except:
            X_data.append(np.zeros(500))
            labels.append('other')

    return clf.predict(X_data)

def classification(vocabulary,clf):
    file_path='../test/images/'

    p_files = input()
    data=[]

    for p in range(int(p_files)):
        file_name =input()
        n_pictures = input()
        for n in range(int(n_pictures)):
            xmin, xmax, ymin, ymax = [int(x) for x in input('').split(' ')]
            img = cv2.imread(file_path+file_name)[int(ymin):int(ymax),int(xmin):int(xmax)]
            data.append({'image': img, 'label': None})

    data=extractingFeautres(vocabulary,data)
    pred=predict(clf,data)
    for i in pred:
        print(i)

    #compareData(readText(),pred)




data_train=load_ClassIdAndCropPhoto('../train/annotations/','../train/images/')

vocabulary=boVW(data_train)

clf=train(extractingFeautres(vocabulary,data_train))

command = input()
if command == 'classify':
    classification(vocabulary,clf)



############additional function##############################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################



def load_ClassIdAndPhoto(annotations_path,photo_path):
    list_names = os.listdir(annotations_path)

    data = []
    for each in list_names:
        tree = ET.ElementTree(file=annotations_path + each)
        root = tree.getroot()
        file_name = root[1].text
        for child in root.iter('object'):
            for bndbox in child.iter("bndbox"):
                class_id=''
                if child[0].text == 'speedlimit':
                    class_id='speedlimit'
                else:
                    class_id='other'
                img=cv2.imread(photo_path+file_name)
                data.append({'image':img,'label':class_id})
    return data

def load_Photo(photo_path):
    list_names = os.listdir(photo_path)

    data = []
    for each in list_names:
        img=cv2.imread(photo_path+each)
        if detect(img)>0.1:
            print(each)
        #detect(img)
        data.append({'image':img,'label':None})
    return data

def load_CropPhoto(annotations_path,photo_path):
    list_names = os.listdir(annotations_path)

    data = []
    for each in list_names:
        tree = ET.ElementTree(file=annotations_path + each)
        root = tree.getroot()
        file_name = root[1].text
        for child in root.iter('object'):
            for bndbox in child.iter("bndbox"):
                img = cv2.imread(photo_path + file_name)[int(bndbox[1].text):int(bndbox[3].text),
                      int(bndbox[0].text):int(bndbox[2].text)]
                data.append({'image': img, 'label': None})
    return data

def detect(image):
    # image = cv2.imread('images/road646.png')

    height, width, channel = image.shape

    add_height = 0
    add_width = 0
    scale = 10
    sq_height = int(height / scale)
    sq_width = int(width / scale)
    ilosc_iteracji = 0
    stepMove = 0.5

    for w in range(scale - 1):
        add_height += sq_height
        for s in range(scale - 1):
            add_width += sq_width
            a = 0
            tmp_h = 0
            tmp_w = 0
            while True:
                data = []
                xmin=0 + add_width - tmp_w
                xmax=sq_width + add_width + tmp_w
                ymin=0 + add_height - tmp_h
                ymax=sq_height + add_height + tmp_h
                img = image[ymin:ymax,xmin:xmax]
                data.append({'image': img, 'label': None, 'bndbox':[xmin,xmax,ymin,ymax]})
                print(predict(clf,extractingFeautres(vocabulary,data)))
                cv2.imshow('s', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ilosc_iteracji += 1
                a += stepMove
                if w - a < 0 or s - a < 0 or w + a > scale - 2 or s + a > scale - 2:
                    break
                tmp_h += math.ceil(stepMove * sq_height)
                tmp_w += math.ceil(stepMove * sq_width)
        add_width = 0
    #print(ilosc_iteracji)

    ile = 0
    a=0
    bnd=[]
    for x in data:
        bnd.append(x['bndbox'])
    d = predict(clf, extractingFeautres(vocabulary, data))
    for i in d:

        if i == 'speedlimit':
            xmin+=data[a]['bndbox'][0]
            xmax+=data[a]['bndbox'][1]
            ymin+=data[a]['bndbox'][2]
            ymax += data[a]['bndbox'][3]
            ile += 1
        a += 1
    #print(ile / len(d))
    return (ile / len(d))

def accuracyScore(labels,pred):
    print(accuracy_score(labels,pred))

def readText():
    data=[]
    with open('../samples.txt') as f:
        comand=f.readline()
        countFiles=f.readline()
        for i in range(int(countFiles)):
            fileName=f.readline()
            countBndbox=f.readline()
            for i in range(int(countBndbox)):
                bndbox=f.readline()
                bnd=[int(bndbox.split(' ')[0]),int(bndbox.split(' ')[1]),
                            int(bndbox.split(' ')[2]),int(bndbox.split(' ')[3])]
                data.append({'name':fileName.split('\n')[0],'bndbox':bnd,
                             'classId':bndbox.split(' ')[4].split('\n')[0]})
    return data
def compareData(dataFromTxt,pred):
    labels=[]
    for dict in dataFromTxt:
        labels.append(dict['classId'])
    print(accuracy_score(labels,pred))




#load_Photo('../test/images/')
