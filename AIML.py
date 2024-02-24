import numpy as np
import matplotlib.pyplot as plt
geometric = []
failure = 0
n=0
p=0.5
while n<10000:
    result = np.random.choice(['success','failure'],p=(p,1-p))
    if result == 'failure':
        failure+=1
    else:
        geometric.append(failure)
        failure = 0
        n+=1
plt.hist(geometric)
plt.axvline(np.mean(geometric),color='red')
plt.show()