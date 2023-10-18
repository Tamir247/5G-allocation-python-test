import numpy as np
M=8            # D2D pair number 
N=3             # total subchannels                 

B1=0.2         # bandwidth MHz                                      
R=600          # cell radius                                        CHANGECY
L=20           # distance D2D pair transmitter and receiver         STANDART 20
PT=0.01        # D2D transmitter power                              CONST
alfa=4         # pathloss factor                                    CONST

SINRTh=np.power(10,4.6/10)                # 2.88403                 CONST

Pt_max=2                                # CUE max transmitter power CONST
B=0.15                                  # subchannel bandwidth      CONST
N0= np.power(10,-90/10)/1000                                       #CONST