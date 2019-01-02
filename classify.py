import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("seña", help="número de la seña en ./señas",
                    type=int)
parser.add_argument("--ignore","-i", action='store_true',help="ignora los videos en los que no el programa funciona correctamente")
args = parser.parse_args()

#número de seña a analizar
fileClassify=int(args.seña)

#Ignora seña si está en la lista de los que no funcionan
fileIgnore=(4,8,9,10,11,14,16,20,21,22,23,24,25,26,30,31,32,33,34,35,36,37,39,42,44,45,47)
if args.ignore and (fileNumber in fileIgnore):
    sys.exit("File ignored")

#seña y (altura, centro, distancia mínima, promedio, distancia recorrida)

def FindDistance(A,B):
 return np.sqrt(np.power((A[0]-B[0]),2) + np.power((A[1]-B[1]),2))

def getInterestFromFile(fileNumber):
    with open('./out2/class%d.txt'%fileNumber) as file:
        name=file.readline()
        line=file.readline()
        hLeft=np.array([float(n) for n in ((line.split('[')[1].split(']')[0]).split())])
        hRight=np.array([float(n) for n in ((line.split('[')[2].split(']')[0]).split())])
        line=file.readline()
        centerLeft=np.array([float(n) for n in ((line.split('[')[1].split(']')[0]).split())])
        centerRight=np.array([float(n) for n in ((line.split('[')[2].split(']')[0]).split())])
        line=file.readline()
        dMinLeft=np.array([float(n) for n in ((line.split('[')[1].split(']')[0]).split())])
        dMinRight=np.array([float(n) for n in ((line.split('[')[2].split(']')[0]).split())])
        line=file.readline()
        avLeft=np.array([float(n) for n in ((line.split('[')[1].split(']')[0]).split())])
        avRight=np.array([float(n) for n in ((line.split('[')[2].split(']')[0]).split())])
        line=file.readline()
        distLeft=np.array([float(n) for n in ((line.split('[')[1].split(']')[0]).split())])
        distRight=np.array([float(n) for n in ((line.split('[')[2].split(']')[0]).split())])

    return name,hLeft,hRight,centerLeft,centerRight,dMinLeft,dMinRight,avLeft,avRight,distLeft,distRight

#distancia de un punto a la cara
def distFace(A):
 return np.sqrt(np.power(A[0],2) + np.power(A[1],2))

#Extrae las trayectorias del archivo
def getTrajectory(fileNumber):
    with open('./out2/traj%d.txt'%fileNumber) as file:
        line=file.readline()
        face=tuple([int(n) for n in (line.split()[0][1:-1],line.split()[1][0:-1])])
        line=file.readline()
        trajLeft=np.array([[int(n) for n in ((line.split('[')[1].split(']')[0]).split())]])
        trajRight=np.array([[int(n) for n in ((line.split('[')[2].split(']')[0]).split())]])
        for line in file:
            trajLeft = np.append(trajLeft, [[int(n) for n in ((line.split('[')[1].split(']')[0]).split())]], 0)
            trajRight = np.append(trajRight, [[int(n) for n in ((line.split('[')[2].split(']')[0]).split())]], 0)

    #cambia referencia de coordenadas en video a  posicion de la cara
    trajLeft[:,0]=trajLeft[:,0]-face[0]
    trajLeft[:,1]=-(trajLeft[:,1]-face[1])
    trajRight[:,0]=trajRight[:,0]-face[0]
    trajRight[:,1]=-(trajRight[:,1]-face[1])
    return trajLeft,trajRight

#extrae puntos de interés a partir de trayectorias de ambas manos
def getInterest(fileNumber):
    trajLeft,trajRight=getTrajectory(fileNumber)
    #Centros de las trayectorias
    xCenterLeft=(max(trajLeft[:,0])-min(trajLeft[:,0]))/2+min(trajLeft[:,0])
    yCenterLeft=(max(trajLeft[:,1])-min(trajLeft[:,1]))/2+min(trajLeft[:,1])
    centerLeft=np.array([xCenterLeft,yCenterLeft])
    xCenterRight=(max(trajRight[:,0])+min(trajRight[:,0]))/2
    yCenterRight=(max(trajRight[:,1])+min(trajRight[:,1]))/2
    centerRight=np.array([xCenterRight,yCenterRight])

    #máxima altura de la trayectoria
    hLeft=trajLeft[np.where(trajLeft[:,0] == min(trajLeft[:,0]))]
    hRight=trajRight[np.where(trajRight[:,0] == min(trajRight[:,0]))]
    hLeft=np.array(hLeft[0])
    hRight=np.array(hRight[0])

    #distancia mínima a la cara
    minLeft=0
    dmin=100000
    for p in trajLeft:
        if abs(distFace(p))<dmin:
            minLeft=p
            dmin=abs(distFace(p))
    minRight=0
    dmin=100000
    for p in trajRight:
        if abs(distFace(p))<dmin:
            minRight=p
            dmin=abs(distFace(p))


    #promedio de las trayectorias
    avxLeft=sum(trajLeft[:,0])/len(trajLeft[:,0])
    avyLeft=sum(trajLeft[:,1])/len(trajLeft[:,1])
    avxRight=sum(trajRight[:,0])/len(trajRight[:,0])
    avyRight=sum(trajRight[:,1])/len(trajRight[:,1])
    avLeft=np.array([avxLeft,avyRight])
    avRight=np.array([avxRight,avyRight])

    #velocidad de las manos
    spxLeft=[(a-b) for a, b in zip(trajLeft[::1,0], trajLeft[1::1,0])]
    spyLeft=[(a-b) for a, b in zip(trajLeft[::1,1], trajLeft[1::1,1])]
    spxRight=[(a-b) for a, b in zip(trajRight[::1,0], trajRight[1::1,0])]
    spyRight=[(a-b) for a, b in zip(trajRight[::1,1], trajRight[1::1,1])]

    #distancia recorrida por las manos
    distLeft=np.array([sum(abs(x) for x in spxLeft),sum(abs(x) for x in spyLeft)])
    distRight=np.array([sum(abs(x) for x in spxRight),sum(abs(x) for x in spyRight)])

    return hLeft,hRight,centerLeft,centerRight,minLeft,minRight,avLeft,avRight,distLeft,distRight

#params=hL,hR,centerL,centerR,dMinL,dMinR,avL,avR,distL,distR
#llena arreglo con parámetros a partir de archivos de entrenamiento
params=np.array([getInterestFromFile(1)[1:]])
names=[getInterestFromFile(1)[0]]
for i,a in enumerate([15,7,18,38,40,3,6,13,43,49,51]):
    params=np.append(params,[getInterestFromFile(a)[1:]],0)
    names.append(getInterestFromFile(a)[0])
seña=np.array(getInterest(fileClassify))

compAll=np.zeros(12)
comp=np.zeros((10, 1))
for i in range(len(params)):
    for j in range(10):
        comp[j]=FindDistance(params[i][j],seña[j])
    compAll[i]=sum(comp)
#print(compAll)
#print(names)
index=np.where(compAll == min(compAll))
guess=names[int(index[0])]
sys.exit(guess)
