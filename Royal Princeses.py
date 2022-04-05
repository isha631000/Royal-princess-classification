#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import cv2
import pandas as pd
import matplotlib 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
img = cv2.imread('C:/ROYAL/diana_test_images/diana_test_2.jpg')
img.shape
plt.imshow(img)


# In[31]:


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray.shape
plt.imshow(gray,cmap='gray')


# In[38]:


face_cascade = cv2.CascadeClassifier('C:/ROYAL/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/ROYAL/opencv/haarcascades/haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces
(x,y,w,h) = faces[0]
x,y,w,h
#the value 255 for red colour
face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),20)
plt.imshow(face_img)


# In[37]:


cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),20)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),20)
        

plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(roi_color, cmap='gray')


# In[39]:


path_to_data = "C:/ROYAL/dataset/"
path_to_cr_data = "C:/ROYAL/dataset/cropped/"

#for accessing the operating system we use import os

import os
img_dirs = []
#accessing the files in dataset iteratively
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)
img_dirs


# In[40]:


#copying and removing of directories and files
import shutil
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data)
#shutil.rmtree is used to remove empty directories
os.mkdir(path_to_cr_data)

cropped_image_dirs = []
celebrity_file_names_dict = {}


# In[42]:


def get_chopped_image_if_2_eyes_found(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = face_img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if(len(eyes)>=2):
            return roi_color
cropped_img = get_chopped_image_if_2_eyes_found('C:/ROYAL/diana_test_images/diana_test_2.jpg')
plt.imshow(cropped_img)


# In[43]:


for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]
    print(celebrity_name)
    celebrity_file_names_dict[celebrity_name] = []
    for entry in os.scandir(img_dir):
        roi_color = get_chopped_image_if_2_eyes_found(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_cr_data + celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ",cropped_folder)
            cropped_file_name = celebrity_name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_file_name 
            
            cv2.imwrite(cropped_file_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count += 1      
            


# In[44]:


import numpy as np
import pywt #Wavelet transform open source software for python
import cv2    

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H


# In[45]:


im_har = w2d(cropped_img,'db1',5)
plt.imshow(im_har, cmap='gray')


# In[47]:


class_dict = {}
count = 1
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1
class_dict


# In[48]:


X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            continue
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name])     


# In[49]:


len(X[0])


# In[50]:


X = np.array(X).reshape(len(X),4096).astype(float)
X.shape


# In[51]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


# In[52]:


print(classification_report(y_test, pipe.predict(X_test)))


# In[53]:


#GRID SEARCH CV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}


# In[54]:


scores = []
best_estimators = {}
import pandas as pd
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[55]:


best_estimators


# In[56]:


best_clf = best_estimators['svm']
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_clf.predict(X_test))
cm


# In[58]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[59]:


import joblib 
# Save the model as a pickle in a file 
joblib.dump(best_clf, 'saved_model.pkl') 


# In[60]:


import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))


# In[ ]:




