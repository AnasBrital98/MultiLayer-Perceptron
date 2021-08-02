import numpy as np
from random import choice

# AND Logic circuit 
AND_data = [(np.array([1,1,1]),1),
                 (np.array([1,0,1]),0),
                 (np.array([0,1,1]),0),
                 (np.array([0,0,1]),0)]   

# OR Logic circuit 
OR_data = [(np.array([1,1,1]),1),
                 (np.array([1,0,1]),1),
                 (np.array([0,1,1]),1),
                 (np.array([0,0,1]),0)]

# XOR Logic circuit 
XOR_data = [(np.array([1,1,1]),0),
                 (np.array([1,0,1]),1),
                 (np.array([0,1,1]),1),
                 (np.array([0,0,1]),0)]

def heaviside(x):
   return 1 if x>= 0 else 0

def Perceptron(training_data,circuit):
    w = np.random.rand(3)
    learning_rate = 0.1
    number_iterations = 100        
    
    for i in range(number_iterations):
      x,y = choice(training_data)
      z = x.dot(w)
      y_predicted = heaviside(z)
      w += learning_rate * (y - y_predicted)*x
    
    print(f"Prediction for {circuit} Logic Circuit :  ")
    for x,y in training_data:
      z = x.dot(w)
      y_pred = heaviside(z)
      print(f"x = {x[:2]} , z = {z} , y_pred = {y_pred}")  
    

Perceptron(AND_data, "AND")

Perceptron(OR_data, "OR")

Perceptron(XOR_data, "XOR")





