# Python_Image-Classifier-with-SVM

__IMPORTANTE__

__Os Códigos que estiverem entre "{}" podem ser usados para testar imagens diferentes das usadas no treino do modelo.__

__A principio, você pode ignorá-los.__

- Importando as bibliotecas necessárias.

      import numpy as np

      import cv2

      from sklearn.svm import SVC

- Criando uma lista de identificação para as imagens.

      Y = []

- A lista será preenchidas com valores de 0 a 1 que indicarão o que represetam. 
- A quantidade de itens a se usar no seu classificador é você quem define, nesse por ser um pequeno
 exemplo só usarei 2; Maçãs e Bananas.
- 0 == Maçã
- 1 == Banana
- No caso do exemplo serão usadas 24 imagens, 12 de Maçãs e 12 de Bananas.

      for h in range(1,25):

            if (h <= 12):

              Y.append(0)

            else:

              Y.append(1)


- Tranformando a lista em array.

      Y = np.array(Y)

- Criando uma lista para guardar as imagens.

      imagens = []

- Preenchendo a lista com as imagens, seguindo a ordem de seleção estabelecido ao preencher Y.
- Seguir a ordem é extremamente importante, perceba que no Y eu coloquei para que ele preenchesse as 12 primeiras posições com 0, ou seja, indiquei que essas 12 posições serão indicadores de maçãs. Sendo assim, na hora de carregar as imagens as 12 primeiras precisam obrigatoriamente ser de maçãs.

      for i in range(1,25):

        imagens.append(cv2.imread('data_'+str(i)+'.png'))

- Tranformando a lista em array.

      X = np.array(imagens)

- Ao fazer isso, sua lista se tornará um array de 4 dimensões.

- Então, temos que transformá-lo em um de 2 dimensões.

      n_samples = len(X)

      data_images = X.reshape((n_samples, -1))

- {

- Seguindo a mesma lógica apresentada acima.

      imgs = []


      for j in range(0, quantidade de imagens a serem analasidas):

        imgs.append(cv2.imread('teste_'+str(j)+'.jpg'))

      T = np.array(imgs)

      t_samples = len(T)

      teste_imagens = T.reshape((t_samples, -1))

- }

- Criando os modelos de treino e de teste.

      from sklearn.model_selection import train_test_split

      X_train, X_test, Y_train, Y_test = train_test_split(data_images,Y, test_size = 0.20, random_state = 0)

- Escalonando os valores.

      from sklearn.preprocessing import StandardScaler

      sc_X = StandardScaler()

      X_train = sc_X.fit_transform(X_train)

      sc_X2 = StandardScaler()

      X_test = sc_X2.fit_transform(X_test)

- {

      sc_T = StandardScaler()

      testes_images = sc_T.fit_transform(teste_imagens)

- }

- Classificador SVM.

      classifier = SVC(kernel = 'rbf', random_state = 0)

- Treinando.

      classifier.fit(X_train,Y_train)

- Predict.

      Y_pred = classifier.predict(X_test)

- {

      Y_pred2 = classifier.predict(teste_imagens)

      # Perceba que, Y_pred2 te retornar os valores (0,0,1,0,1,1)
      
      # Ou seja, as imagens carregadas, teste_0, teste_1 e teste_2 [...] correspondem a:
      
      # teste_0 == 0 | Maca
      
      # teste_1 == 0 | Maca
      
      # teste_2 == 1 | Banana
      
      # teste_1 == 0 | Maca
      
      # teste_2 == 1 | Banana
      
      # teste_2 == 1 | Banana      

- }

- Confusion Matrix.

      from sklearn.metrics import confusion_matrix

      cm = confusion_matrix(Y_test, Y_pred)



