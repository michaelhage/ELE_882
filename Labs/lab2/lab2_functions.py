import numpy as np

W = np.random.random([3,3])
F = np.random.random([10,10])

a,b = W.shape
m,n = F.shape

if b % 2 == 1:
    sr = int((b-1)/2)
    sl = int(sr)

else:
    sr = int(b/2)
    sl = int(sr - 1)

if a % 2 == 1:
    sd = int((a - 1)/ 2)
    su = int(sd)

else:
    sd = int(a / 2)
    su = int(sd - 1)

F_temp = np.zeros([m+sl+sr,n+su+sd])

for i in range(sr,m):
    for j in range(su,n):
        print(i,j)
        F_temp[i-sr, j-su] = np.sum(F[i-sr:i+sl+1, j-su:j+sd+1]*W)