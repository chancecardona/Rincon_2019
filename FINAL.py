import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.signal import max_len_seq
sys.path.insert(0,"/home/intern/Desktop/Record/SOAPY/SoapySDR/build/python3")
from CalibrationFuncs import *
from SoapySDR import *
import SoapySDR as sp
print(sp)

center = 915e6
bw = 40e6
N = 2**15
frames = 100
upsample = 32
numCal = 100

def switch(d, on):
    d.writeGPIODir('MAIN', 0xFF)
    if on:
        GPIOVal = 0x01
    else:
        GPIOVal = 0x00
    d.writeGPIO('MAIN', GPIOVal)

def transmitSignal():
    signal = max_len_seq(int(np.log2(N)),length = N)[0]
    return signal 



def main(args):
    #Set Constants
    results = sp.Device.enumerate()
    sdrs = [sp.Device(results[i]) for i in range(len(results))]
    args = dict(serial='0009081C05C11822')
    d = sp.Device(args)

    #Setttings, create streams
    for i in sdrs:
        for j in range(2):
            i.setAntenna(SOAPY_SDR_RX, j, 'LNAW')
            i.setFrequency(SOAPY_SDR_RX, j, center)
            i.setGain(SOAPY_SDR_RX, j, 30)
            i.setSampleRate(SOAPY_SDR_RX, j, bw)
            #i.setBandwidth(SOAPY_SDR_RX, j, 50e6)
            print(i.getFrequency(SOAPY_SDR_RX,j))
            print(i.getSampleRate(SOAPY_SDR_RX,j))
    streams = [i.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16,[0,1]) for i in sdrs]
    ret_vals = [sdrs[i].activateStream(streams[i]) for i in range(len(sdrs))]

    #Define buffers
    buffers_0 = [np.zeros(2*N).astype(np.int16) for i in range(len(sdrs))]
    buffers_1 = [np.zeros(2*N).astype(np.int16) for i in range(len(sdrs))]
    collectionC = np.zeros([len(sdrs)*2, numCal*N*2]).astype(np.int16)
    collection = np.zeros([len(sdrs)*2, frames*N*2]).astype(np.int16)
    old_times = np.zeros(len(sdrs))



    #Begin calibration
    switch(d, True) #Toggles GPIO 
    
    #letting everything settle. Throwaway first 5000 reads
    time.sleep(1)
    for i in range(int(5e3)):
        ret_vals = [sdrs[j].readStream(streams[j],[buffers_0[j],buffers_1[j]],N) for j in range(len(sdrs))]
    print('Beginning calibration...')

    for i in range(numCal):
        ret_vals = [sdrs[j].readStream(streams[j],[buffers_0[j],buffers_1[j]],N) for j in range(len(sdrs))]
        collectionC[::2,i*2*N:(i+1)*N*2] = np.array(buffers_0)
        collectionC[1::2,i*2*N:(i+1)*N*2] = np.array(buffers_1)
    
    switch(d, False)
    time.sleep(.25)

    #Start recording data
    for i in range(frames):
        ret_vals = [sdrs[j].readStream(streams[j],[buffers_0[j],buffers_1[j]],N) for j in range(len(sdrs))]
        #print(all([i.ret == N for i in ret_vals]))
        collection[::2,i*2*N:(i+1)*N*2] = np.array(buffers_0)
        collection[1::2,i*2*N:(i+1)*N*2] = np.array(buffers_1)
        #new_times = np.array([i.timeNs for i in ret_vals])
        #SR = 1./((new_times-old_times)/(N * 1e9))
        #print(SR)
        #old_times = new_times
   
    print('Done recording')



    #Collect data, close streams
    A = (collectionC[:,0::2] + collectionC[:,1::2]*1j)[:, 80*N:81*N]
    B = (collection[:,0::2] + collection[:,1::2]*1j)[:, 80*N:90*N]
    rawCal = A
    raw = B

    ret_vals = [sdrs[i].deactivateStream(streams[i]) for i in range(len(sdrs))]
    ret_vals = [sdrs[i].closeStream(streams[i]) for i in range(len(sdrs))]
    peaks = np.zeros(A.shape[0] - 1)
   

    #Finds delays in calibration
    for i in range(1, A.shape[0]):
        #Finds each channels delay relatiive to channel 0
        peaks[i-1] = findDelay(A[0,:], A[i,:], upsample) % N
        intDelay = int(peaks[i-1])# - 5
       
    print('CAL  PEAKS:', peaks)


    #Time align
    peaksA = peaks
    for i in range(1, B.shape[0]):
        intDelay = int(peaks[i-1]) #- 5
	#time syncs calibration signal
        A[i,:] = np.roll(A[i,:], intDelay)
        A[i,:] = fracDelay(A[i,:], peaks[i-1] - intDelay)
	#time syncs data
        B[i,:] = np.roll(B[i,:], intDelay)
        B[i,:] = fracDelay(B[i,:], peaks[i-1] - intDelay)
        
    
    
    #plt.figure()
    #Normalizes each channel
#    for i in range(B.shape[0]):
#        plt.plot(B[i,:] * 200 / np.linalg.norm(B[i,:]), label=str(i))
#    plt.legend()


    #Phase alignment
    rot = phaseDelay2(A) #get phases from calibration signal
    C = np.diag(rot).dot(B) #use them to phase align data

    np.save('rawCal', rawCal)
    np.save('cal', A)
    np.save('rawData', raw)
    np.save('finalData', C)


    #Real and Imag of Data
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.real(C.T))
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.subplot(2,1,2)
    plt.plot(np.imag(C.T))
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()

    #Abs
    plt.figure()
    plt.plot(np.abs(C.T))
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    plt.show()
   




if __name__=='__main__':
    args = None
    main(args)
