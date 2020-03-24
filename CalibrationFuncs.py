import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sig,Amplitude,tShift):
    return Amplitude*np.exp(-np.power((x-tShift) - mu, 2.) / (2 * np.power(sig, 2.)))

def nuttal(N,center, W):
    t = np.linspace(-N//2,N//2,N+1)
    a0 = 0.355768
    a1 = 0.487396
    a2 = 0.144232
    a3 = 0.012604
    N = 2*W

    x = a0 + a1*np.cos(2*np.pi*(t-center)/(N-1))+ a2*np.cos(4*np.pi*(t-center)/(N-1))+ a3*np.cos(6*np    .pi*(t-center)/(N-1))
    x[np.abs(t - center) > W] = 0
    
    return x

def overlapAndSave(x, h):
    L = len(x)
    N = len(h)
    P = 4*N
    nseg = (L+N-1)//(P-N+1) + 1
    x = np.concatenate((np.zeros(N-1), x, np.zeros(P)))
    xp = np.zeros((nseg, P), dtype=np.complex64)
    yp = np.zeros((nseg, P), dtype=np.complex64)
    y = np.zeros(nseg*(P-N+1), dtype=np.complex64)

    for p in range(nseg):
        xp[p, :] = x[p*(P-N+1):p*(P-N+1)+P]
        yp[p, :] = np.fft.ifft(np.fft.fft(xp[p, :]) * np.fft.fft(h, P))
        y[p*(P-N+1):p*(P-N+1)+P-N+1] = yp[p, N-1:]
    y = y[N//2:N//2+L]
    return y

def fracDelay(x, delay):
    N = 2**7
 
    #sinc MUST be critically sampled.
    t = np.linspace(-N//2,N//2,N+1)
    sinc = np.sinc((t-delay))
    
    #windowing for better frequency response
    nut_width = 50
    nut = nuttal(N,delay,nut_width)
    
    #plt.figure()
    #plt.plot(nut, label='Nuttal Window')
    #plt.plot(sinc, label='Sinc Filter')
    #plt.xlabel('Samples')
    #plt.title('Fractional Delay Filter Components')
    #plt.legend()


    h = sinc * nut

    #plt.figure()
    #plt.plot(10*np.log(np.abs(np.fft.fft(h, len(h)*100))))
    #plt.title('FFT of Filter')
    #plt.ylabel('dB')
    #plt.xlabel('Freq (Hz)')

    out = overlapAndSave(x, h)
    return out

def findDelay(x1, x2, upsample):
    #correlate signals. Need to be same size
        #this one could take dif sizes... acts weird.
    #y = np.fft.fftshift(overlapAndSave(x1, x2))
    #y = np.correlate(x1, x2, 'same')
    #plt.plot(np.abs(x1))
    #plt.plot(np.abs(x2))
    y = np.fft.fftshift(np.fft.fft(x1) * np.fft.fft(x2).conj())


    #upsample
    y2 = np.concatenate([
        np.zeros([int(np.round(len(y)*(upsample-1)/2.0))]),
        y,
        np.zeros([int(np.round(len(y)*(upsample-1)/2.0))])
    ])
    
    #
    y3 = np.abs(np.fft.ifft(np.fft.fftshift(y2)))
    
    #plt.figure()
    #plt.plot(y3)
    #plt.title('Upsampled Correlation')

    #finds peaks
    peak = np.argmax(y3)/upsample
    return peak
    


#Done with comparing angles. Input the array with all the channels
def phaseDelay1(A):
    B = np.mean(A/A[0,:], axis=1)
    #C = np.diag(B.conj()).dot(A)
    return B.conj()

#Done with SVD
def phaseDelay2(A):
    u,s,v = np.linalg.svd(A.dot(A.T.conj()))
    rot = v[0,:]
    #D = np.diag(rot).dot(A)
    return rot











