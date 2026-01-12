import numpy as np
class PositionalEncoding:
    def Calculate(self,embed):
        length,m=embed.shape
        positions=np.arange(length)[:,np.newaxis]
        dimension=np.arange(m)[np.newaxis:]
        rate=1/np.power(10000,(2*(dimension//2))/m)
        radians=positions*rate
        posEn=np.zeros_like(embed)
        posEn[:,0::2]=np.sin(radians[:,0::2])
        posEn[:,1::2]=np.cos(radians[:,1::2])
        return posEn