from  sklearn.linear_model  import  SGDClassifier
from  sklearn.preprocessing  import  LabelEncoder
from  sklearn.model_selection  import  train_test_split
from  simplepreprocessor  import  SimplePreprocessor
from  simpledatasetloader import  SimpleDatasetLoader
from  imutils  import  paths

imagePaths  =  list(paths.list_images("datasets/animals"))

sp  =  SimplePreprocessor(32,  32)

sdl  =  SimpleDatasetLoader(preprocessors=[sp])

(data,  labels_0)  =  sdl.load(imagePaths,  verbose=500)

data  =  data.reshape((data.shape[0],  3072))

le  =  LabelEncoder()

labels  =  le.fit_transform(labels_0)

(trainX,  testX,  trainY,  testY)  =  train_test_split(data,  labels,test_size=0.25,  random_state=42)


for r in (None,  "l1",  "l2"):
    print("[INFO]  training  model  with  `{}`  penalty".format(r))
    model  =  SGDClassifier(loss="log",  penalty=r,  max_iter=10,
                            learning_rate="constant",  tol=1e-3,  eta0=0.01,random_state=12)

    model.fit(trainX,  trainY)
    acc  =  model.score(testX,  testY)
    print("[INFO]  `{}`  penalty  accuracy:  {:.2f}%".format(r,acc  *  100))


    
