
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math



arrays = np.array([[7.420,8.690,9.960,7.500,7.420],[4,4,3,4,4],[2,4,2,2,1]])
arrays= arrays.transpose()


print(arrays)
y = [133,122.5,122.5,122.15,114.1]
w = np.array([0,0,0])
b = 0
def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost


def compute_gradient(X, y, w, b): 
    m,n = X.shape      
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w = w_in  
    b = b_in
    
    for i in range(num_iters):

        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history



w,b,J_hist= gradient_descent(arrays,y,w,b,compute_cost,compute_gradient,120e-4,1500000)

print(np.dot(np.array([7.6,4,2]),w)+b)
