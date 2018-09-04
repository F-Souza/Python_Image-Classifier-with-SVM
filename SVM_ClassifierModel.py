import numpy as np
import cv2
from sklearn.svm import SVC

Y = []

for h in range(1,25):
    if (h <= 12):    
        Y.append(0)
    else:
        Y.append(1)

Y = np.array(Y)

imagens = []
for i in range(1,25):
    imagens.append(cv2.imread('data_'+str(i)+'.png'))
X = np.array(imagens)
n_samples = len(X)
data_images = X.reshape((n_samples, -1))

"""
imgs = []
for j in range(0,Quantidade de Imagens a serem analisadas):
    imgs.append(cv2.imread('teste_'+str(j)+'.jpg'))
T = np.array(imgs)
t_samples = len(T)
teste_imagens = T.reshape((t_samples, -1))
"""

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data_images,Y, test_size = 0.20, random_state = 0)

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