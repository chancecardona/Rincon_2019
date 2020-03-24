import numpy as np
import matplotlib.pyplot as plt
from CalibrationFuncs import *
import sys

f = sys.argv[1]
di = 'data' + f + '/'
A = np.load(di+'finalData.npy')
B = np.load(di+'rawData.npy') 
C = np.load(di+'rawCal.npy')
D = np.load(di+'cal.npy')
upsample = 32

def plot(data):
    for i in range(data.shape[0]):
        plt.subplot(4,1,1)
        plt.plot(np.abs(data[i, :]), label=str(i))
        plt.subplot(4,1,2)
        plt.plot(np.real(data[i, :]))
        plt.subplot(4,1,3)
        plt.plot(np.imag(data[i, :]))
    plt.subplot(4,1,4)
    #rot = phaseDelay2(data)
    #print(np.angle(rot/rot[0]))
    d = []
    n = int(10)
    B = data.reshape(data.shape[0],int(data.shape[1]/n),n)
    for i in range(int(n)):
        t = phaseDelay2(B[:,:,i])
        d.append(np.angle(t/t[0]))
    d = np.array(d)
    plt.plot(d)
    plt.legend()



##Plots recorded data
#plt.figure()
#for i in range(A.shape[0]):
#    A[i, :] = A[i, :] / np.linalg.norm(A[i, :])
#plot(A)
#plt.legend()



#Plots data before phase shift
plt.figure()
#timeshifts
for i in range(1, B.shape[0] - 1):
    peak = findDelay(B[0,:], B[i,:], upsample)
    intDelay = int(peak)
    B[i, :] = np.roll(B[i,:], intDelay)
    B[i, :] = fracDelay(B[i,:], peak - intDelay)
#phaseshifts
for i in range(B.shape[0]):
    B[i, :] = B[i, :] / np.linalg.norm(B[i, :])
plot(B)
plt.suptitle('Data before Phase Shift')


plt.figure()
rot = phaseDelay2(D)
B = np.diag(rot).dot(B)    
for i in range(B.shape[0]):
    B[i, :] = B[i, :] / np.linalg.norm(B[i, :])
plot(B)
plt.suptitle('Data After Phase Shift')





#Plots recorded cal signal (synced in time. Before phase shift)
plt.figure()
for i in range(D.shape[0]):
    D[i, :] = D[i, :] / np.linalg.norm(D[i, :])
plot(D)
plt.suptitle('Calibration Signal Before Phase Shift')


#rot = phaseDelay2(D)
D = np.diag(rot).dot(D)
#Plots data after phase shift
plt.figure()
for i in range(D.shape[0]):
    D[i, :] = D[i, :] / np.linalg.norm(D[i, :])
plot(D)
plt.suptitle('Calibration Signal After Phase Shift')


plt.show()
