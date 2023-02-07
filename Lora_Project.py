#imports
from __future__ import division
import numpy as np
from scipy.signal import chirp,correlate,correlation_lags
import matplotlib.pyplot as plt
from cmath import phase
from cmath import exp,pi       
import random
import math
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
import statistics

##########################Functions###########################

##Whitening/ Dewhitening

def whitening (message, S):
    message = [int(x) for x in message]
    S = [int(x) for x in S]
    r= (len(message)//len(S))
    
    W = []
    while len(message)>len(S):
        m = message[:len(S)]
        W = W + list(a^b for a,b in zip(m,S))
        message = message[len(S):]
        
    if len(message)<=len(S):
        W = W + list(a^b for a,b in zip(message,S))
        
    W = ''.join(str(x) for x in W)
    return W
    
  

def dewhitening(message,S):
    message = [int(x) for x in message]
    S = [int(x) for x in S]
    r= round(len(message)//len(S))
    
    W = []
    while len(message)>len(S):
        m = message[:len(S)]
        W = W + list(a^b for a,b in zip(m,S))
        message = message[len(S):]
        
    if len(message)<=len(S):
        W = W + list(a^b for a,b in zip(message,S))
        
    W = ''.join(str(x) for x in W)
    return W

##Channel Coding/ Decoding

def hamming_encode(s):
    K=4
    # Read in K=4 bits at a time and write out those plus parity bits
    code =""
    add = 0
    while len(s) >= K:
        nibble = s[0:K]
        code += hamming(nibble)
        s = s[K:]
        if len(s)<K and len(s) != 0:
            
            for i in range(len(s), K):
                s +="0"
                add +=1
    return code,add

def hamming(bits):
    K=4
    # Return given 4 bits plus parity bits for bits (1,2,3), (2,3,4) and (1,3,4)
    t1 = parity(bits, [0,1,3])
    t2 = parity(bits, [0,2,3])
    t3 = parity(bits, [1,2,3])
    return t1 + t2 + bits[0] + t3 + bits[1:] #again saying, works only for 7,4

def parity(s, indicies):
    # Compute the parity bit for the given string s and indicies
    sub = ""
    for i in indicies:
        sub += s[i]
    return str(str.count(sub, "1") % 2)

def  hamming_decode(xx):
    message = xx[0]
    subs = xx[1]
    message = [int(x) for x in message]
    W = ""
    while len(message)>= 7:
        
        s = message[:7]
        """  Hamming  decoding  of the 7 bits  signal  """
        b1= (s[0]+s[2]+s[4]+s[6]) % 2
        b2= (s[1]+s[2]+s[5]+s[6]) % 2
        b3= (s[3]+s[4]+s[5]+s[6]) % 2
        b=4*b3+2*b2+b1 
        # the  integer  value
        if b==0 or b==1 or b==2 or b==4:
            W += ''.join(map(str,(s[2],s[4],s[5] ,s[6])))
            #W += [s[2],s[4],s[5] ,s[6]]
        else:
            y=[s[0],s[1] ,s[2],s[3],s[4] ,s[5],s[6]]
            y[b-1]=(s[b -1]+1) % 2   # correct  bit b
            W += ''.join(map(str,(s[2],s[4],s[5] ,s[6])))
            #W += [s[2],s[4],s[5] ,s[6]]
        message = message[7:]

    W = W[:len(W)-subs]
    return W

##Interleaving/ De-Interleaving

def to_nm_list(A, n, m):
    S = [[0 for i in range(m)] for j in range(n)]
    if len(A) == n*m:
        c=0
        for k in range(n):
            for i in range(m):
                S[k][i] = A[c]
                c+=1
    elif len(A) < n*m:
        B = ['0' for i in range(n*m - len(A))]
        b=''
        b=b.join(B)
        A+=b
        S = to_nm_list(A, n, m)
    else:
        S=[]
        k=len(A)//(n*m)
        for i in range(k):
            M=A[i*(n*m):((i+1)*(n*m))]
            S.append(to_nm_list(M, n, m))
        A = A[k*(m*n):]
        if A != '':
            S.append(to_nm_list(A, n, m))
    return (S)

def interleaving(A, sf, cr):
    """"the interlaeving block 
        sf:Spreading Factor 7,12
        cr:Coding Rate 1,4
        #input A : SF x (4+CR) matrix
        #output: CR+4 x SF matrix
                                
                                """
    O=''
    p=[]
    A=to_nm_list(A, sf, cr+4)
    #print(A)
    if len(A[1]) == 4+cr :
        S = [[0 for i in range(sf)] for j in range(4+cr)]
        for i in range(0, 4+cr):
            for j in range(0, sf):
                k =sf-np.mod((i-j), sf) - 1
                l =cr+3-i
                S[i][j] = A[k][l]
        #O=O.join(sum(S, []))
        for i in S:
            for j in i:
                O+=j
    else:
        for i in A:
            I=''
            #print(i)
            #I=I.join(sum(i, [])
            for j in i:
                     for k in j:
                         I+=k
            #print(I)
            p.append(interleaving(I, sf, cr))
            #print(p)
        O=O.join(p)
    return(O)

def deinterleaving(A, sf, cr):
    O=''
    p=[]
    A=to_nm_list(A, cr+4, sf)
    if len(A[1]) == sf :
        S = [[0 for i in range(4+cr)] for j in range(sf)]
        for i in range(0, sf):
            for j in range(0, cr+4):
                l =np.mod((sf+cr+4-j-i), sf)
                k =cr+3-j
                S[-i][j] = A[k][l]
        #O=O.join(sum(S, []))
        for i in S:
            for j in i:
                O+=j
    else:
        for i in A:
            I=''
            #I=I.join(sum(i, []))
            for h in i:
                for j in h:
                    I+=j
       
            
            p.append(deinterleaving(I, sf, cr))
        O=O.join(p)
    return(O)

def BER (imsg,smsg):

    BER=0

    for i in range(0,min(len(imsg),len(smsg))):
        
            if imsg[i] != smsg[i]:
             BER=BER+1
    return(BER)

##Modulation/ Demodulation

def gray2dec(num):
    """retourne le nombre entier correspondant au code Gray num"""
    shift = 1
    while True:
        idiv = num >> shift
        num ^= idiv
        if idiv <= 1 or shift == 32: 
            return num
        shift <<= 1

def str_to_sym(str,SF):
    list=[]
    for i in range(len(str)//SF):
        a=str[i:i+SF]
        b=gray2dec(int(a,2))
        list.append(b)
    
    
    return list

def dec2gray(num):
    """retourne le code gray correspondant au nombre entier 'num' """
    return (num>>1)^num
 
dec2bin = lambda x, n=8: bin(x)[2:].zfill(n)

def LoRa_Modulation(SF, symbols, sign):
    
    """ INPUTS :

   SF : Spreading Factor (7:12)
   symbols : vector of symbols to modulate
   sign : (-1,+1), used to choose between down-chirps and up-chirps.
   
OUTPUT :

   txSig : IQ modulated signal"""


    M = 2**SF  # Number of possible symbols
    ka = np.arange(1, M+1)
    

    fact1 = np.exp(1j*sign*np.pi*(ka**2)/M) # Compute it only one time
    r = 0
    txSig = np.zeros((M, len(symbols)), dtype=complex)  # Preallocation
    for k in range(len(symbols)):
        symbK = fact1 * np.exp(2j*np.pi*(symbols[k]/M)*ka)
        txSig[:,r] = symbK
        r += 1
    txSig = txSig.flatten()
    txSig[0] = 1
    """plt.plot(txSig)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()"""
    return list(txSig)

def LoRa_modulator(SF,stream,sign):
    signal=[]
    for h in range(len(stream)//SF):
        symbol=str_to_sym(stream[h*SF:SF*(h+1)],SF)
        signal=signal+LoRa_Modulation(SF, symbol, sign)
    if (len(stream))%SF!=0:
        symbol=str_to_sym(stream[(len(stream)//SF)*SF:len(stream)]+'0'*(SF-len(stream)%SF),SF)
        signal=signal+LoRa_Modulation(SF, symbol, sign)
    
    return signal,len(stream)

def tfc(a,SF):
    M=2**SF
    a = np.array(a)
    
    b = np.array(LoRa_Modulation(SF,[0],1))
    correlation=circular_correlation(a, b)
    lags=[i for i in range(M)]
    if lags[np.argmax(abs(correlation))]==0:
        return(0)
    
    lag=lags[M-np.argmax(abs(correlation))]
    
    return(lag)

def Demodulation(s,SF,longueur):
    
    M=2**SF
    a=len(s)//M
    
    demodulated_sequence=''
    
    for i in range(0,a):
       
        h=s[i*M:(i+1)*M]
        demodulated_sequence+=str(dec2bin(dec2gray(tfc(h,SF)),SF))
    
        
    return demodulated_sequence#[0:longueur]

def circular_correlation(x, y):
    """
    Calculates the circular correlation of two signals.
    """
    x_fft = np.fft.fft(x)
    y_fft = np.fft.fft(y)
    xy_corr = np.fft.ifft(x_fft * np.conj(y_fft))
    return np.real(xy_corr)

##Channel Simulation/ Synchronization

def awgn(s,SNRdB):
    
    #AWGN channel
    #Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    #returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    #Parameters:
       # s : input/transmitted signal vector
        #SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        #r : received signal vector (r=s+n)
    
    gamma = 10**(SNRdB/10) #SNR to linear scale
    
    a=[i.real for i in s]
    b=[i.imag for i in s]

    P=sum(np.abs(a)**2)/len(a)+1j*sum(np.abs(b)**2)/len(b) #Actual power in the vector
    
    #print('Actual power in the vector: '+P)
    
    N0=P/gamma # Find the noise spectral density
    #print('Noise spectral density: '+N0)

    ss=np.append([0 for i in range(random. randint(1, 10))],s, axis=None)

    if isrealobj(ss):# check if input is real/complex object type
       n = sqrt(N0/2)*standard_normal(np.shape(ss)) # computed noise
    else:
       n = sqrt(N0/2)*(standard_normal(np.shape(ss))+1j*standard_normal(np.shape(ss)))
 
    r = ss + n # received signal
    print(sum(abs(np.array(s))**2)/len(s),sum(abs(n)**2)/len(n))

    return r

def synchro(r, pream):
    sync=[]
    a=len(pream)
    correlation = correlate(r, pream, mode="full")# Function to calculate cross-correlation,
    lags = correlation_lags(len(r), len(pream), mode="full")
    lag = lags[np.argmax(correlation)]# extract the best matching shift and then shift
    print(f"Best lag: {lag}")

    r = r[lag:]
    sync=r[a:len(r)]

    return (r)


#Chaîne de transmission

def TX(S,whitening_sequence,preamble,SF):

    #print('initial signal: '+ S)
    whitened_signal=whitening(S,whitening_sequence)
    #print('whitened_signal: '+ whitened_signal)
    coded_signal,j=hamming_encode(whitened_signal)#[0],hamming_encode(whitened_signal)[1]
    #print('j: '+ str(j))
    #print('coded_signal: '+ coded_signal)
    interleaved_signal=interleaving(coded_signal, 7, 4)
    #print('interleaved_signal: '+ interleaved_signal)
    modulated_signal=preamble+LoRa_modulator(SF,interleaved_signal,1)[0]
    #print('longueur du signal émis: '+ str(len(modulated_signal)))
    return modulated_signal,j,S,whitening_sequence,LoRa_modulator(SF,interleaved_signal,1)[1]

#Passage par le canal
def channel(S,SNR,preamble):
    R=awgn(S,SNR)
    #print('longueur du signal reçu: '+ str(len(R)))
    synchronized_signal=synchro(R,preamble)
    #print('longueur du signal synchronisé: ' + str(len(synchronized_signal)))
    return synchronized_signal

#Chaîne de réception
def RX(S,whitening_sequence,message,j,SF,longueur):
    
    
    demodulated_signal=Demodulation(S,SF,longueur)[4*SF:len(Demodulation(S,SF,longueur))+1]
    #print('signal démodulé: '+ demodulated_signal)
    deinterleaved_signal= deinterleaving(demodulated_signal,7,4)
    #print('deinterleaved signal: '+ deinterleaved_signal)
    decoded_signal= hamming_decode((deinterleaved_signal,j))
    #print('signal décodé: '+ decoded_signal)
    signal_post_traitement=dewhitening(decoded_signal,whitening_sequence)
    #print('signal_post_traitement: '+ signal_post_traitement)
    #print('#################BER###################\n'+ str(BER(message,signal_post_traitement))+ ' bits erronés parmis '+ str(len(message))+' \nBER est donc: '+ str(BER(message,signal_post_traitement)/len(message)))
    return(BER(message,signal_post_traitement)/len(message))

def LoRa_Communication_Chain_Simulator(message,SNR,preamble,whitening_sequence,SF):
    modulated_signal,j,S,whitening_sequence,longueur=TX(message,whitening_sequence,preamble,SF)

    return RX(channel(modulated_signal,SNR,preamble),whitening_sequence,message,j,SF,longueur)



def test(message,SNR,preamble,whitening_sequence,SF):
    #1000 samples
    observation=[]
    for i in range(1000):
        observation.append(LoRa_Communication_Chain_Simulator(message,SNR,preamble,whitening_sequence,SF))
    return statistics.mean(observation)

def rand_key(p):
	key1 = ""
	for i in range(p):
		temp = str(random.randint(0, 1))
		key1 += temp
		
	return(key1) 

def BER_SNR_fig(message,preamble,whitening_sequence,SF):
    #SNR changes from -28 to 4
    y=[]
    x=[]
    for x_SNR in range(-28,5):
        y.append(test(message,x_SNR,preamble,whitening_sequence,SF))
        x.append(x_SNR)
 
    return x,y











x_7,y_7=BER_SNR_fig('10010100101010010100101110101',LoRa_Modulation(7,[0],1)*2+LoRa_Modulation(7,[0],-1)*2,'1011',7)
x_8,y_8=BER_SNR_fig('10010100101010010100101110101',LoRa_Modulation(8,[0],1)*2+LoRa_Modulation(8,[0],-1)*2,'1011',8)
x_9,y_9=BER_SNR_fig('10010100101010010100101110101',LoRa_Modulation(9,[0],1)*2+LoRa_Modulation(9,[0],-1)*2,'1011',9)
x_10,y_10=BER_SNR_fig('10010100101010010100101110101',LoRa_Modulation(10,[0],1)*2+LoRa_Modulation(10,[0],-1)*2,'1011',10)                                   

plt.plot(x_7,y_7,label="SF=7")
plt.plot(x_8, y_8, label="SF=8")
plt.plot(x_9, y_9, label="SF=9")
plt.plot(x_10, y_10, label="SF=10")
plt.xlabel('SNR(dB)')
plt.ylabel('BER')
plt.semilogy()
plt.legend()
plt.grid(True)
plt.show()
