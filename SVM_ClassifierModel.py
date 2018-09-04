import numpy as np
import cv2
from sklearn.svm import SVC

Y = []

for h in range(1,176):
    if (h <= 25):    
        Y.append(0)
    elif (h > 25 and h <= 50):
        Y.append(1)
    elif h > 50 and h <= 75:
        Y.append(2)
    elif h > 75 and h <= 100:
        Y.append(3)
    elif h > 100 and h <= 125:
        Y.append(4)
    elif h > 125 and h <= 150:
        Y.append(5)
    elif h > 150 and h <= 175:
        Y.append(6)


Y = np.array(Y)

imagens = []
for i in range(1,176):
    imagens.append(cv2.imread('data ('+str(i)+').jpg'))
X = np.array(imagens)
n_samples = len(X)
data_images = X.reshape((n_samples, -1))

"""
imgs = []
for j in range(0,3):
    imgs.append(cv2.imread('teste_'+str(j)+'.jpg'))
T = np.array(imgs)
t_samples = len(T)
teste_imagens = T.reshape((t_samples, -1))
"""

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data_images,Y, test_size = 0.40, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
sc_X2 = StandardScaler()
X_test = sc_X2.fit_transform(X_test)

"""
sc_T = StandardScaler()
teste_imagens = sc_T.fit_transform(teste_imagens)
"""

classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

"""
Y_pred2 = classifier.predict(teste_imagens)
"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)