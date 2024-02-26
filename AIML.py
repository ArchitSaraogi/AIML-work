
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv('NIFTY50_all.csv')

# Specify the headers of the columns you want to convert
column_headers = ['Open', 'High', 'Low','Close']

# Convert the specified columns into NumPy arrays
arrays = [0,0,0,0]
for i in range(len(column_headers)):
    arrays[i] = np.array(df[column_headers[i]],dtype=np.float64)
x_1_train = arrays[0]
x_2_train = arrays[1]
x_3_train = arrays[2]
y = arrays[3]


def calulate_model(x,w,b):
    f_n= np.dot(x,w) +b
    return f_n
def J(y,f_n):
    J_ouput =0
    for i in range(len(y)):
        J_output+= (math.pow((f_n-y[i]),2))/(2*len(y))
    return J_output
w = [0,0,0]
b = 0
def changed_w(f_n,y,w,b,alpha,x_1_train,x_2_train,x_3_train):

    dJ_dw1=0
    dJ_dw2=0
    dJ_dw3=0
    dJ_db=0
    for i in range(len(y)):
        dJ_dw1+= (f_n -y[i])*x_1_train[i]/len(x_3_train)
        dJ_dw2+= (f_n -y[i])*x_2_train[i]/len(x_3_train)
        dJ_dw3+= (f_n -y[i])*x_3_train[i]/len(x_3_train)
        dJ_db+= (f_n -y[i])/len(x_3_train)
    w[0] -= alpha * dJ_dw1
    w[1] -= alpha * dJ_dw2
    w[2] -= alpha * dJ_dw3
    b-=dJ_db

    return [b,w]


def calulate_weights(iterations,y,b,w,alpha,x_1_train,x_2_train,x_3_train):
    set = []
    for i in range(len(x_1_train)):
        row= []
        row.append(x_1_train[i])
        row.append(x_2_train[i])
        row.append(x_3_train[i])
        set.append(row)

    for i in range(iterations):
        b=changed_w(calulate_model(np.array(set[i],dtype=np.float64),np.array(w,dtype=np.float64),b),y,np.array(w,dtype=np.float64),b,alpha,x_1_train,x_2_train,x_3_train)[0]
        w=changed_w(calulate_model(np.array(set[i],dtype=np.float64),np.array(w,dtype=np.float64),b),y,np.array(w,dtype=np.float64),b,alpha,x_1_train,x_2_train,x_3_train)[1]
    return [b,w]
    
b=calulate_weights(100,y,b,w,0.1,x_1_train,x_2_train,x_3_train)[0]
w = np.array(calulate_weights(100,y,b,w,0.01,x_1_train,x_2_train,x_3_train)[1])
l=[22290.00,22297.50,22186.10]
final = b+ np.dot(w,l)
print(final)


    
        









