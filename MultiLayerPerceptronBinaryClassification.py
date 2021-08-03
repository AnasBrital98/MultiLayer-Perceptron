import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs,make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class NeuralNetwork:
    
    def __init__(self,x,y,layers,learning_rate,n_iterations):
        self.x = x.T
        self.y = y.reshape((1,y.shape[0]))
        self.layers = layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.params = {}
        for i in range(1,len(self.layers)):
            self.params['w'+str(i)] = np.random.randn(self.layers[i],self.layers[i-1])
            self.params['b'+str(i)] = np.random.randn(self.layers[i],1)
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def dsigmoid(self,x):
        return x*(1-x)
    
    def forward_propagation(self,x):
        activations = {'A0':x}
        for i in range(1 , (len(self.params) // 2) +1):
            z = np.dot( self.params['w'+str(i)] , activations['A'+str(i-1)]) + self.params['b'+str(i)]
            activations['A'+str(i)] = self.sigmoid(z)
        return activations
    
    def gradients(self,activations):
        grads = {}
        dz = activations['A'+str(len(self.params) // 2)] -self.y
        for i in reversed(range(1,(len(self.params) // 2)+1)):
            grads['dw'+str(i)] = 1 / self.y.shape[1] * np.dot(dz,activations['A'+str(i-1)].T)
            grads['db'+str(i)] = 1 / self.y.shape[1] * np.sum(dz,axis=1,keepdims=True)
            dz = np.dot(self.params['w'+str(i)].T , dz) * self.dsigmoid(activations['A'+str(i-1)]) 
        return grads
    
    def back_propagation(self,activations):
        grads = self.gradients(activations)
        for i in range(1,len(self.params)//2) :
            self.params['w'+str(i)] -= self.learning_rate * grads['dw'+str(i)]
            self.params['b'+str(i)] -= self.learning_rate * grads['db'+str(i)]
    def train(self):
        for i in range(self.n_iterations):
            activations = self.forward_propagation(self.x)
            self.back_propagation(activations)
        
    def prediction(self,x):
       activations = self.forward_propagation(x)
       return activations['A'+str(len(self.params) // 2)]
        
    def predict(self,x):
       activations = self.forward_propagation(x)
       predictions = activations['A'+str(len(self.params) // 2)]
       for i in range(0,len(predictions[0])):
           if predictions[0,i] > 0.5 :
               predictions[0,i] = 1
           else :
               predictions[0,i] = 0               
       return  predictions
   
    def displayResult(self):
      fig, ax = plt.subplots()
      ax.scatter(self.x[0, :], self.x[1, :], c=self.y, cmap='bwr', s=50)
      x0_lim = ax.get_xlim()
      x1_lim = ax.get_ylim()
    
      resolution = 100
      x0 = np.linspace(x0_lim[0], x0_lim[1], resolution)
      x1 = np.linspace(x1_lim[0], x1_lim[1], resolution)
      X0, X1 = np.meshgrid(x0, x1)
      XX = np.vstack((X0.ravel(), X1.ravel()))
      y_pred = self.prediction(XX)
      y_pred = y_pred.reshape(resolution, resolution)
    
      ax.pcolormesh(X0, X1, y_pred, cmap='bwr',alpha = 0.3 , zorder = -1)
      ax.contour(X0, X1, y_pred, c='g')
      plt.show()
                  

x, y = make_blobs(n_samples=200 , n_features=2 , centers=2 , random_state=0) 
#make_moons(n_samples=200, noise=0.2, random_state=0)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.5,random_state=1)



print("DataSet Shape : ")
print(f"x_train.shape => {x_train.shape}")
print(f"y_train.shape => {y_train.shape}")
print(f"x_test.shape => {x_test.shape}")
print(f"y_test.shape => {y_test.shape}")

NN = NeuralNetwork(x_train, y_train, layers=(x.shape[1],32,32,1), learning_rate=0.1, n_iterations=10000)
NN.train()
NN.displayResult()

predictions = NN.predict(x_test.T)
y_hat_Array = [int(i) for i in predictions[0]]


#Evaluate The Model
print("\n\nModel Evaluation  : ")
myconfusionMatrix = confusion_matrix(y_test , y_hat_Array)

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
count = np.count_nonzero(match)    

print(f"Number of samples That The Model was able to classify successfully in The Test Data : {count}")
print(f"Number of samples that The Model Failed to classify in The Test Data is : {len(predictions[0])-count}")

