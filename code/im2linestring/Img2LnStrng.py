#import Libraries
import numpy as np
import cv2
import csv
import sys
import os.path

if(len(sys.argv) != 9):
	print('wrong number of input arguments')
	print('Usage Img2LnStrng.py minLineLength maxLineGap numIntersections accuracy grouping')
	print('Suggestion: Img2LnStrng.py 20 3 150 .8 30 gtf_out.csv Vegas.png')


minLineLength = int(sys.argv[1])
maxLineGap = int(sys.argv[2])
numIntersections = int(sys.argv[3])
accuracy = float(sys.argv[4])
grouping = float(sys.argv[5])
outFile = sys.argv[6]
inpFile = sys.argv[7]
resultImage = sys.argv[8]

#import image
#img = cv2.imread('CurveTry.tif')
img = cv2.imread(inpFile)

#covert image to grayscale and then extract edges
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)

#Hough Line parameters
#minLineLength = 20
#maxLineGap = 10

#Creating line segments out of Image
lines = cv2.HoughLinesP(gray,accuracy,np.pi/180,numIntersections,minLineLength,maxLineGap)

myData = [['ImageId', 'WTxtd']]
i = 1;

#Initialize point vector
points = [[0,1,[0]]]
for x1,y1,x2,y2 in lines[0]:
	addPoint1 = True
	addPoint2 = True
	#Check to see if a similar point is already in the vector
	for checking in points:
		if(abs(x1-checking[0])+abs(y1-checking[1]) < grouping):
			addPoint1 = False
			connection1 = points.index(checking)
	for checking in points:
		if(abs(x2-checking[0])+abs(y2-checking[1]) < grouping):
			addPoint2 = False
			connection2 = points.index(checking)
	#if statement that adds 2 pts, 1 pt, or just connections
	#print(addPoint1,addPoint2)
	if(addPoint1 and addPoint2):
		points.append([x1,y1,[i+1]])
		points.append([x2,y2,[i]])
		i +=2
	elif(addPoint1 and (not addPoint2)):
		points.append([x1,y1,[i]])
		points[connection2][2].append(i)
		i +=1
	elif((not addPoint1) and addPoint2):
		points.append([x2,y2,[i]])
		points[connection1][2].append(i)
		i += 1
	else:
		points[connection1][2].append(connection2)
		points[connection2][2].append(connection1)
	#print(points)

#For loop goes through all connections and 
#writes a LINESTRING command for each segment
#It also draws the line on the original image 
for x, y, connection in points:
	for z in connection:
		command = "LINESTRING(%d %d , %d %d)" % (x, y, points[z][0], points[z][1])
		cv2.line(img,(x,y),(points[z][0],points[z][1]),(0,255,0),2)		
		myData.append(['ImageId',command])

#Create image with lines drawn on it
completeImgName = os.path.join('./ParameterTesting/Img',resultImage)
cv2.imwrite(completeImgName,img)

#Create CSV file
completeCSVName = os.path.join('./ParameterTesting/CSV',outFile)
myFile = open(completeCSVName,'w')
with myFile:
	writer = csv.writer(myFile)
	writer.writerows(myData)

#Reference code for possible future steps
#Finds Eularian path
"""
# Finding Eulerian path in undirected graph
# Przemek Drochomirecki, Krakow, 5 Nov 2006

def eulerPath(graph):
    # counting the number of vertices with odd degree
    odd = [ x for x in graph.keys() if len(graph[x])&1 ]
    odd.append( graph.keys()[0] )

    if len(odd)>3:
        return None
    
    stack = [ odd[0] ]
    path = []
    
    # main algorithm
    while stack:
        v = stack[-1]
        if graph[v]:
            u = graph[v][0]
            stack.append(u)
            # deleting edge u-v
            del graph[u][ graph[u].index(v) ]
            del graph[v][0]
        else:
            path.append( stack.pop() )
    
    return path
"""
