import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


x , y = make_blobs(n_samples=100,n_features=2,centers=2,random_state=0)

y = y.reshape(y.shape[0],1)


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def cross_entropy(y,a):
    return -(y*np.log(a) + (1-y)*np.log(1-a))/y.shape[0]

def initialize_w_b(x):
    w,b = np.random.randn(x.shape[1],1) , np.random.randn(1)
    return w,b
def forward_propagation(x,w,b):
    z = x.dot(w) + b
    a = sigmoid(z)
    return a
def gradients(x,a,y):
    dw , db = np.dot(x.T,a-y) / x.shape[0] , (a-y)/x.shape[0]
    return dw,db

def backward_propagation(x,w,b,a,y,learning_rate):
    dw , db = gradients(x, a, y)
    w , b = w - learning_rate * dw , b - learning_rate *db
    return w,b
def predict(x,w,b):
    a = forward_propagation(x, w, b)
    return a >= 0.5
def perceptron(x,y,learning_rate=0.2,n_iter=1000):
    w,b = initialize_w_b(x)
    
    losses = []
    
    for i in range(n_iter):
        a = forward_propagation(x, w, b)
        loss = cross_entropy(y, a)
        losses.append(loss)
        w , b = backward_propagation(x, w, b, a, y, learning_rate)
        
    y_predicted = predict(x, w, b)

    return y_predicted , losses , w , b

def displayResult(losses):
    fig , ax = plt.subplots()
    x_lim = ax.get_xlim()
    x = np.linspace(x_lim[0] , x_lim[1] , 1000)
    losses = np.sum(losses , axis = 1)
    losses = np.array(losses)
    plt.plot(x , losses)
    plt.show()
    

y_pred , losses , w , b = perceptron(x, y)

displayResult(losses)
