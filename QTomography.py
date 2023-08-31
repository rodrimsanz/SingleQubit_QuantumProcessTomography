import numpy as np
from scipy.linalg import inv
import pandas as pd
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cmath

#we define the pauli matrices

I = [[1, 0], [0, 1]]
X = [[0, 1], [1, 0]]
Y = [[0, -1j], [1j, 0]]
Z = [[1, 0], [0, -1]]
sigma = np.array([I, X, Y, Z])

#defining now the states

zero = [1,0]
one = [0,1]
plus = [(1/2)**(1/2),(1/2)**(1/2)]
plus_i = [(1/2)**(1/2), ((1/2)**(1/2))*1j]
phi = np.array([zero, one, plus, plus_i])

#computing now the sigma_i*phi_k states

sphi = np.array([])

for p in phi:
    for s in sigma:
        result = np.dot(s, p)
        sphi = np.append(sphi, result)


sigmaphi = sphi.reshape((16, 2))

#defining now the measurement bra's:

m1_zero = [1,0]
m2_zero = [(1/2)**(1/2),(1/2)**(1/2)]
m3_zero = [(1/2)**(1/2), ((1/2)**(1/2))*1j] 
M_l = np.array([m1_zero, m2_zero, m3_zero])

#computation of the C_ni and C_nj factors

C = np.array([])

for l in M_l:
    for k in sigmaphi:
        result = np.dot(np.conjugate(l), k)
        C = np.append(C, result)

C_lki = np.reshape(C, (3,4, 4))

c_kli = np.transpose(C_lki, (1,0,2))

Cni = np.reshape(c_kli, (12,4))


#computation of Anm

Anm = np.zeros((12,16),complex)
n = 0
while n < 12:
    i = 0
    k = 0
    while i < 4:
        j = 0
        while j < 4:
            Anm[n][k] = Cni[n][i] * np.conjugate(Cni[n][j])
            j = j + 1
            k = k + 1
        i = i + 1
    n = n+1

#we now add to the Anm matrix 4 rows regarding the conditions in order to have trace 1
v1 = [0,1,0,0,1,0,0,0,0,0,0,complex(0,-1),0,0,complex(0,1),0]
v2 = [0,0,1,0,0,0,0,complex(0,1),1,0,0,0,0,complex(0,-1),0,0]
v3 = [0,0,0,1,0,0,complex(0,-1),0,0,complex(0,1),0,0,1,0,0,0]
v4 = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
Anm = np.vstack((Anm, v1, v2, v3,v4))

#now our matrix is 16x16 with rank = 16. We compute the inverse now
Anm_inv = np.round(np.linalg.inv(Anm),3)

#we import our data from the file and obtain the average of all measurements to end up with 1 only vector: mean_P
#IMPORTANT: it depends on the order of the data within the measurements file

file_dat = 'Path_of_the_data_file'
data = np.loadtxt(file_dat)

vectorP = [[] for _ in range(len(data))]
for j in range(len(data)):
    for i in range(len(data[0])):
     if (i%4) == 2:
            vectorP[j].append(data[j][i])

mean_P = np.reshape(np.mean(vectorP, axis=0),(4,3))
mean_P[[2, 3]] = mean_P[[3, 2]]

mean_P_vector = mean_P.flatten()

#we now add the elements from the additional 4 equations in order to have trace 1

mean_P_vector = np.concatenate((mean_P_vector,[0,0,0,1]))
print(mean_P_vector)

#we obtain the process matrix now

Chi = np.round(np.reshape(np.dot(Anm_inv,mean_P_vector),(4,4)),3)
print(Chi)


#plot3d of the process matrix

x = np.arange(4)
y = np.arange(4)
Xaxis, Yaxis = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
modulo = np.abs(Chi)
theta = np.angle(Chi)

fase = np.angle(Chi.flatten())

cmap = plt.cm.get_cmap('coolwarm')  # Elige un colormap, por ejemplo 'cool'
normalize = plt.Normalize(min(fase), max(fase))
colors = cmap(normalize(fase))

bars = ax.bar3d(Xaxis.flatten(), Yaxis.flatten(), np.zeros_like(modulo).flatten(), 1, 1, modulo.flatten(), color = colors)

ax.set_xlabel('Column')
ax.set_ylabel('Row')
ax.set_zlabel('Modulus')

plt.show()

#Closest unitary

eigenvalues, eigenvectors = np.linalg.eig(Chi)

e_v0 = np.array(eigenvectors[0], dtype=complex)
e_v1 = np.array(eigenvectors[1], dtype=complex)
e_v2 = np.array(eigenvectors[2], dtype=complex)
e_v3 = np.array(eigenvectors[3], dtype=complex)


proyector0 = np.outer(e_v0, np.conj(e_v0))
proyector1 = np.outer(e_v1, np.conj(e_v1))
proyector2 = np.outer(e_v2, np.conj(e_v2))
proyector3 = np.outer(e_v3, np.conj(e_v3))

Mmixed = eigenvalues[0]*proyector0 + eigenvalues[1]*proyector1 + eigenvalues[2]*proyector2 + eigenvalues[3]*proyector3

#Closest unitary process:
print(np.round(proyector0,3))

#'Noisy' mixed process
print(np.round(Mmixed,5))
#Noise rate:
print(1-eigenvalues[0]^2)

