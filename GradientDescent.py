import matplotlib.pyplot as plt
import numpy as np

learning_rate = 0.1
nbr_iterations = 100
threashHold = 1e-6

def display(g,x,y,step):
    x = np.arange(x,y,step)
    y = g(x)
    plt.plot(x,y, "r-.")
    plt.show()


def f(x):
    return x**2 - x -1
def df(x):
    return 2*x - 1

def gradient(f,df,x0):
    gr = df(x0)
    x = np.arange(-10,10,0.01)
    plt.plot(x , f(x))
    i = 0
    while i < nbr_iterations and abs(gr) > threashHold:
        plt.scatter(x0, f(x0), color='red')
        x0 -= learning_rate * gr
        gr = df(x0)
        i+=1
    plt.show()    
    return x0    


print(gradient(f, df, 6))
display(f, -10, 10, 0.01)