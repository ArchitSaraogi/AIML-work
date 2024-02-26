import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv('NIFTY50_all.csv')

# Specify the headers of the columns you want to convert
column_headers = ['Open', 'High', 'Low','Close']

# Convert the specified columns into NumPy arrays
arrays = [0,0,0]
for i in range(len(column_headers)-1):
    arrays[i] = np.array(df[column_headers[i]][-15:],dtype=np.float64)

arrays = np.array(arrays)/100
arrays = arrays.transpose()
y = np.array(df[column_headers[len(column_headers)-1]],dtype=np.float64)/100


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
w=np.zeros(3)
b =0

l=[22290.00,22297.50,22186.10]
l = np.array(l)/100
print(l)
w,b,J_hist= gradient_descent(arrays,y,w,b,compute_cost,compute_gradient,5e-3,15000)
print(np.dot(l,w)+b)


    
        









