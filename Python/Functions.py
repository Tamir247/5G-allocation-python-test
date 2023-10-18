from GlobalVariables import *
import numpy as np
import itertools as tl

def inter_DUEtoCUE_(r_DTUE):
    global PT, alfa
    return np.array([PT/np.power((r_DTUE),alfa)])[0]

def inter_DUEtoDUE_(x_DTUE, x_DRUE, y_DTUE, y_DRUE, m):
    global PT, alfa
    #DECLERATION
    inter_DUEtoDUE = np.zeros((m,m))
    #CALCULATION
    for i, j in list(tl.product(range(m), range(m))):
        inter_DUEtoDUE[i,j] = PT/np.power(np.sqrt((x_DTUE[j]-x_DRUE[i])**2+(y_DTUE[j]-y_DRUE[i])**2),alfa)
    for i in range(m):
        inter_DUEtoDUE[i,i] = 0
    return inter_DUEtoDUE

def package_(pg, n):
    #DECLERATION
    Package = {}                    #return type
    #CALCULATION
    for i in range(n):
        x = np.where(pg == i)[0]
        Package[i] = x.tolist()
    return Package

def JudgeCUEPower(inter_DUEtoCUE, package, r_CUE, n):
    global  alfa, N0, SINRTh, B
    #DECLARATION
    CUE_Interference = 0
    CUE_Pt = np.zeros((n))          #return type
    CUE_Capacity = np.zeros((n))    #return type
    #CALCULATION
    for i in range(n):
        CUE_Interference = np.sum(inter_DUEtoCUE[package[i]])
        CUE_Pt[i] = (CUE_Interference + N0) * SINRTh * np.power(r_CUE[i], alfa)
        CUE_Capacity[i] = B*np.log2(1+SINRTh)

    if np.max(CUE_Pt)>Pt_max:
        raise Exception()

    return CUE_Pt, CUE_Capacity 

def inter_CUEtoDUE_(r_DTUE, package,x_CUE,x_DRUE,y_CUE,y_DRUE,r_CUE,  m, n):
    global  alfa
    #DECLERATION
    inter_CUEtoDUE = np.zeros((m, n))#return type
    
    #CALCULATION
    CUE_Pt = JudgeCUEPower(inter_DUEtoCUE_(r_DTUE), package, r_CUE, n=n)[0]
    for i, j in list(tl.product(range(m),range(n))):
        inter_CUEtoDUE[i,j] = CUE_Pt[j] / np.power(np.sqrt((x_CUE[j]-x_DRUE[i])**2+(y_CUE[j]-y_DRUE[i])**2), alfa)

    return inter_CUEtoDUE

def PackageCapacity_DUE(inter_DUEtoDUE, inter_CUEtoDUE, r_DRUE_shift, Package ,n, m):
    global alfa, N0, PT, B1
    #DECLERATION
    DUE_SINR = np.zeros((m))
    DUE_Interference = np.zeros((m))
    DUE_Capacity = np.zeros((m))
    PackageCapacity_DUE = np.zeros((n)) #return type
    #CALCULATION
    for i in range(n):
        Capacitytemp=0
        for kk in range(len(Package[i])):
            index = Package[i][kk]
            DUE_Interference[index] = sum(inter_DUEtoDUE[index,Package[i]]) + inter_CUEtoDUE[index, i]
            DUE_SINR[index] = PT/np.power(r_DRUE_shift[index],alfa)/(DUE_Interference[index]+N0)
            DUE_Capacity[index] = B1*np.log2(1+DUE_SINR[index])
            Capacitytemp += DUE_Capacity[index]
        PackageCapacity_DUE[i]=Capacitytemp

    return PackageCapacity_DUE


def total_(package, r_CUE, r_DTUE,r_DRUE_shift, x_DTUE, x_DRUE, y_DTUE, y_DRUE, x_CUE, y_CUE, due_n, cue_n):
    return np.sum(JudgeCUEPower(inter_DUEtoCUE_(r_DTUE), package, r_CUE, cue_n)[0]) + np.sum(PackageCapacity_DUE(inter_DUEtoDUE_(x_DTUE, x_DRUE, y_DTUE, y_DRUE, due_n), inter_CUEtoDUE_(r_DTUE, package,x_CUE,x_DRUE,y_CUE,y_DRUE,r_CUE, due_n, cue_n),r_DRUE_shift, package , cue_n, due_n))


def details(N, M, CUE, DRUE, DTUE):
    r_CUE = np.array([np.linalg.norm(i) for i in CUE], dtype=np.float32)
    r_DTUE = np.array([np.linalg.norm(i) for i in DRUE], dtype=np.float32)
    r_DRUE_shift = np.array([np.linalg.norm(DRUE[i] - DTUE[i]) for i in range(len(DRUE))], dtype=np.float32)
    return [r_CUE, r_DTUE, r_DRUE_shift, DTUE.T[0], DRUE.T[0], DTUE.T[1], DRUE.T[1], CUE.T[0], CUE.T[1], M, N]