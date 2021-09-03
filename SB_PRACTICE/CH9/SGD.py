from  sklearn.model_selection  import  train_test_split
from  sklearn.metrics  import  classification_report
from  sklearn.datasets  import  make_blobs
import  matplotlib.pyplot  as  plt
import numpy as np

def sigmoid_activation(x):
    return  1.0  /  (1  +  np.exp(-x))

def  sigmoid_deriv(x):
    return  x  *  (1  -  x)


def predict(X,W):
    preds  =  sigmoid_activation(X.dot(W))

    preds[preds<=0.5]=0
    preds[preds>0.5]=1

    return preds

def next_batch(X,y,batchSize):
    for i in np.arange(0,X.shape[0],batchSize):
        yield(X[i:i+batchSize],y[i:i+batchSize])


(X,  y)  =  make_blobs(n_samples=1000,  n_features=2,  centers=2,cluster_std=1.5,  random_state=1)

y  =  y.reshape((y.shape[0],  1))

X  =  np.c_[X,  np.ones((X.shape[0]))]


(trainX,  testX,  trainY,  testY)  =  train_test_split(X,  y,test_size=0.5,  random_state=42)


epochs=100
alpha=0.01
batch_size=32


print("[INFO]  training...")
W  =  np.random.randn(X.shape[1],  1)
losses  =  []


for epoch in np.arange(0,epochs):
    epochLoss  =  []
    for  (batchX,  batchY)  in  next_batch(trainX,  trainY,  batch_size):
        preds  =  sigmoid_activation(batchX.dot(W))
        error  =  preds  -  batchY
        epochLoss.append(np.sum(error  **  2))

        d  =  error  *  sigmoid_deriv(preds)
        gradient  =  batchX.T.dot(d)

        W+=-alpha*gradient
    loss  =  np.average(epochLoss)
    losses.append(loss)

    if  epoch  ==  0  or  (epoch  +  1)  %  5  ==  0:
        print("[INFO]  epoch={},  loss={:.7f}".format(int(epoch  +  1),loss))

print("[INFO]  evaluating...")
preds  =  predict(testX,  W)
print(classification_report(testY,  preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:,  0],  testX[:,  1],  marker="o",  c=testY[:,  0],  s=30)


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,  epochs),  losses)
plt.title("Training  Loss")
plt.xlabel("Epoch  #")
plt.ylabel("Loss")


plt.show()

        

