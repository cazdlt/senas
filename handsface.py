import numpy as np
import cv2
import face_recognition
import argparse
import sys
from matplotlib import pyplot as plt

#low_purple=np.array([120,50,50])
#high_purple=np.array([150,255,255])

#colors
low_pink=np.array([135,50,50])
high_pink=np.array([170,255,255])
low_yellow=np.array([20,50,50])
high_yellow=np.array([40,255,255])
low_lightblue=np.array([80,50,50])
high_lightblue=np.array([105,255,255])
low_purple=np.array([120,50,50])
high_purple=np.array([160,255,255])
low_blue=np.array([98,80,110])
high_blue=np.array([110,255,255])
low_white=np.array([0,0,170])
high_white=np.array([180,20,255])

#argumentos para automatizar proceso
parser = argparse.ArgumentParser()
parser.add_argument("seña", help="número de la seña en ./señas",
                    type=int)
parser.add_argument("--ignore","-i", action='store_true',help="ignora los videos en los que no el programa funciona correctamente")
args = parser.parse_args()


#número de seña a analizar
fileNumber=int(args.seña)

#Ignora seña si está en la lista de los que no funcionan
fileIgnore=(4,8,9,10,11,14,16,20,21,22,23,24,25,26,30,31,32,33,34,35,36,37,39,42,44,45,47)
if args.ignore and (fileNumber in fileIgnore):
    sys.exit("File ignored")


cap = cv2.VideoCapture('./señas/Señas%d.mov'%fileNumber)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
w = int(cap.get(3)/2)
h = int(cap.get(4)/2)
out = cv2.VideoWriter('./out2/path%d.avi'%(fileNumber),fourcc, 10, (w,h))

#filters background noise using morph transformations
def filterBackgroundNoise(mask):
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation = cv2.dilate(mask,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation3,5)
    blur = cv2.GaussianBlur(median,(5,5),0)
    ret,thresh = cv2.threshold(blur,127,255,0)
    return thresh, median


#finds biggest object with hue in [lowColor highColor] inside frame
def findHand(frame,lowColor,highColor):
    blur = cv2.GaussianBlur(frame,(5,5),0)
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv,lowColor,highColor)

    thresh,median=filterBackgroundNoise(mask2)

    cannyO,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (122,122,0), 3)

    max_area=100
    ci=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i

    if len(contours)>0:
        cnts = contours[ci]

        #find convex hull
        hull = cv2.convexHull(cnts)

        #find center of mass of convex hull
        moments = cv2.moments(cnts)
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00
        centerMass=(cx,cy)

        #draw bounding rectangle and convex hull
        x,y,w,h = cv2.boundingRect(cnts)
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(frame,[hull],-1,(255,255,255),2)

    #cv2.imshow('hand',frame)
    return cx,cy;

#colors to find
if fileNumber<=20:
    print("Amarillo - Azul claro")
    low_color1=low_lightblue
    low_color2=low_yellow
    high_color1=high_lightblue
    high_color2=high_yellow
elif fileNumber in (21,22,23,24,27,28,29,37,38,39,40,41,42,43,48):
    print("Morado - Azul")
    low_color1=low_blue
    low_color2=low_purple
    high_color1=high_blue
    high_color2=high_purple
elif fileNumber in (49,50,51,52):
    print("Amarillo - Morado")
    low_color1=low_purple
    low_color2=low_yellow
    high_color1=high_purple
    high_color2=high_yellow
else:
    print("Morado - Blanco")
    low_color1=low_white
    low_color2=low_purple
    high_color1=high_white
    high_color2=high_purple

# Create a mask image for drawing purposes
color = np.random.randint(0,255,(100,3))
ret,frame = cap.read()
frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
mask3 = np.zeros_like(frame)

#find initial hand position
cx1,cy1=findHand(frame,low_color1,high_color1)
cx2,cy2=findHand(frame,low_color2,high_color2)
centerMass=[(cx1,cy1),(cx2,cy2)]

#finds and draws face in initial frame
rgb_frame = frame[:, :, ::-1]
faces  = face_recognition.face_locations(rgb_frame, model="cnn")
if not faces:
    sys.exit("No hay caras.")
for face in faces:
    (top, right, bottom, left) = face
    centerFace=(np.round(left+(right-left)/2).astype(int),np.round(top+(bottom-top)/2).astype(int))
mask3 = cv2.rectangle(mask3, (left, top), (right, bottom), (0, 0, 255), 2)
mask3 = cv2.circle(mask3,centerFace,2,(0, 0, 255),2)


#trajectory arrays for classification
traj=np.array([centerMass])
sp=np.array([[(0,0),(0,0)]])

i=0
while(1):
    ret,frame = cap.read()

    if ret==True:

        frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

        #find hand centers for tracking
        cx1,cy1=findHand(frame,low_color1,high_color1)
        cx2,cy2=findHand(frame,low_color2,high_color2)
        cmOld=centerMass
        centerMass=[(cx1,cy1),(cx2,cy2)]

        #draw the tracks
        for i,(new,old) in enumerate(zip(centerMass,cmOld)):
            mask3 = cv2.line(mask3, new, old, color[i].tolist(), 2)
            frame = cv2.circle(frame,new,5,color[i].tolist(),2)
        img = cv2.add(frame,mask3)

        #adds point to trajectory array
        traj = np.append(traj, [centerMass], 0)
        speed=[np.subtract(centerMass[0],cmOld[0]),np.subtract(centerMass[1],cmOld[1])]
        sp = np.append(sp, [speed], 0)

        #show results
        #cv2.imshow('flow',img)
        out.write(img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    else:
        break

#export trajectory and speed plots
plt.figure(1)
plt.plot(traj[:,0,0],-traj[:,0,1],label="left hand")
plt.plot(traj[:,1,0],-traj[:,1,1],label="right hand")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1,  borderaxespad=0.)
plt.plot(centerFace[0],-centerFace[1],'rs')
plt.title('position')
plt.savefig('./out2/position%d.png'%(fileNumber))


plt.figure(2)
plt.plot(sp[:,0,0],-sp[:,0,1],label='left hand')
plt.plot(sp[:,1,0],-sp[:,1,1],label='right hand')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1,  borderaxespad=0.)
plt.title('speed')
plt.savefig('./out2/speed%d.png'%(fileNumber))

with open('./out2/traj%d.txt'%(fileNumber),"w") as fp:
    fp.write(str(centerFace)+'\n')
    fp.write('\n'.join('{} {}'.format(x[0],x[1]) for x in traj))

cv2.destroyAllWindows()
cap.release()
out.release()

sys.exit("OK")
