'''
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

y_t = [[1,0,0,1],[1,0,1,1]]
y_pr= [[1,0,0,1], [1,0,0,1]]


f1_scor_l = []
for i,j in zip(y_t, y_pr):
    f1_scor = f1_score(i, j, average='macro', zero_division=0)
    f1_scor_l.append(f1_scor)

print(f1_scor_l)'''


import numpy as np
a = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
b = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
print(a)
print('----------------------------')
a_trans = a.transpose()
print(a_trans)
print('----------------------------')
b_trans = b.transpose()
print(b_trans)
print('----------------------------')

print('----------------------------')
b_list = b_trans.tolist()
print(b_list)
print('----------------------------')
a_list = a_trans.tolist()
print(a_list)
print('----------------------------')



from sklearn.metrics import f1_score
f1_l = []
for i,j in zip(a_list,b_list):
    f1 = f1_score(i, j, average='macro', zero_division=0)
    f1_l.append(f1)

print(f1_l)