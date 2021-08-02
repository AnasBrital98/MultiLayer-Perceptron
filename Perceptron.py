import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class Perceptron : 
    
    def __init__(self,learning_rate = 0.1 , number_iter = 1000):
        self.learning_rate = learning_rate
        self.number_iter   = number_iter
        
    
    def fit(self,x,y):
        self.x = x
        self.y = y
    
    
    def initParameters(self , x):
        w = np.random.randn(1,x.shape[1])         
        return w
    
    def heaviside(self,x):
        return 1 if x>= 0 else 0
    
    def heavisideArray(self,x):
        a = [1 if x1>= 0 else 0 for x1 in x[0]]
        return a
    
    def train(self):
        self.w = self.initParameters(self.x)
        
        for i in range(self.number_iter):
            for x , y in zip(self.x , self.y):
                z = np.dot(self.w , x)
                y_hat = self.heaviside(z)
                self.w += self.learning_rate * (y - y_hat) * x
        #self.displayModel()        
                
                
    def predict(self,x):
        z = np.dot(self.w , x)
        a = self.heavisideArray(z)
        return a
    
    def displayModel(self):
        fig , ax = plt.subplots(figsize=(10,7))
        ax.scatter(self.x[:,0] , self.x[:,1] , c = self.y , cmap="bwr")
        x1 = np.linspace(-15,4,100)
        x2 = (-self.w[0][0] * x1 - self.w[0][2]) / self.w[0][1]
        ax.plot(x1,x2 , c='g' , lw=8)
        


x , y = make_blobs(n_samples=200 , n_features=2 , centers=2 , random_state= 0)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.5 , random_state=0)


b = np.ones(x_train.shape[0])
b = b.reshape(b.shape[0] , 1)
x_train = np.hstack((x_train , b))

b = np.ones(x_test.shape[0])
b = b.reshape(b.shape[0] , 1)
x_test  = np.hstack((x_test , b))

print("DataSet Shape : ")
print(f"x_train.shape => {x_train.shape}")
print(f"y_train.shape => {y_train.shape}")
print(f"x_test.shape => {x_test.shape}")
print(f"y_test.shape => {y_test.shape}")



perceptron = Perceptron()
perceptron.fit(x_train, y_train)
perceptron.train()


predictions = perceptron.predict(x_test.T)

print("predictions.shape : ",predictions.shape)
"""
#Evaluate The Model
myconfusionMatrix = confusion_matrix(y_test , predictions)

accuracy = (myconfusionMatrix[0][0] + myconfusionMatrix[1][1]) / np.sum(myconfusionMatrix)
precision = myconfusionMatrix[0][0] / (myconfusionMatrix[0][0] + myconfusionMatrix[0][1])
recall = myconfusionMatrix[0][0] / (myconfusionMatrix[0][0] + myconfusionMatrix[1][0])
specificity = myconfusionMatrix[1][1] / (myconfusionMatrix[1][1] + myconfusionMatrix[0][1])
f1score = 2 * myconfusionMatrix[0][0] / (2 * myconfusionMatrix[0][0] + myconfusionMatrix[1][0] + myconfusionMatrix[0][1])


print(f"Accuracy : {accuracy} .")
print(f"precision : {precision} .")
print(f"recall : {recall} .")
print(f"specificity : {specificity} .")
print(f"f1score : {f1score} .")
 

match = predictions == y_test
SuccessfullyClassified = np.count_nonzero(match)
FailedCalssified = len(predictions) - SuccessfullyClassified

print(f"Number of samples That The Model was able to classify successfully in The Test Data : {SuccessfullyClassified} .")
print(f"Number of samples that The Model Failed to classify in The Test Data is  : : { FailedCalssified } .")

    

"""





