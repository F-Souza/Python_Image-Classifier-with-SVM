# Simple-Image-Classifier-with-SVM

__IMPORTANTE__

__O Código que estiver entre "{}" chaves, é o código a se usar para testar com imagens diferentes das usadas para treinar.__

__A principio, você pode ignorá-los.__

- Importando as bibliotecas necessárias.

      import numpy as np

      import cv2

      from sklearn.svm import SVC

- Criando uma lista de identificacao para as imagens

      Y = []

- A lista será preenchidas com valores de 0 a 6 que indicarão o que represetam.
- 0 == Right
- 1 == Left
- 2 == Close
- 3 == Front
- 4 == If
- 5 == Loop
- 6 == Back
- No caso do exemplo serão usadas 175 imagens.

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

- Tranformando a lista em array.

      Y = np.array(Y)

- Criando uma lista para guardar as imagens.

      imagens = []

- Preenchendo a lista com as imagens, seguindo a ordem de seleção estabelecido ao preencher Y.

      for i in range(1,176):

        imagens.append(cv2.imread('data ('+str(i)+').jpg'))

- Tranformando a lista em array.

      X = np.array(imagens)

- Ao fazer isso, sua lista se tornará um array de 4 dimensões.

- Então, temos que transformá-lo em um de 2 dimensões.

      n_samples = len(X)

      data_images = X.reshape((n_samples, -1))

- {

- Seguindo a mesma lógica apresentada acima.

      imgs = []

- Nesse caso testaremos apenas com 3 imagens.

      for j in range(0,3):

        imgs.append(cv2.imread('teste_'+str(j)+'.jpg'))

      T = np.array(imgs)

      t_samples = len(T)

      teste_imagens = T.reshape((t_samples, -1))

- }

- Criando os modelos de treino e de teste.

      from sklearn.model_selection import train_test_split

      X_train, X_test, Y_train, Y_test = train_test_split(data_images,Y, test_size = 0.40, random_state = 0)

- Escalonando os valores.

      from sklearn.preprocessing import StandardScaler

      sc_X = StandardScaler()

      X_train = sc_X.fit_transform(X_train)

      sc_X2 = StandardScaler()

      X_test = sc_X2.fit_transform(X_test)

- {

      sc_T = StandardScaler()

      testes_images = sc_T.fit_transform(testes_image)

- }

- Classificador SVM.

      classifier = SVC(kernel = 'rbf', random_state = 0)

- Treinando.

      classifier.fit(X_train,Y_train)

- Predict.

      Y_pred = classifier.predict(X_test)

- {

      Y_pred2 = classifier.predict(testes_images)

- }

- Confusion Matrix.

      from sklearn.metrics import confusion_matrix

      cm = confusion_matrix(Y_test, Y_pred)



