#import Libraries
import numpy as np
import cv2
import csv


outFile = 'gtf_out.csv'
inpFile = 'gt.tif'

#import image
#img = cv2.imread('CurveTry.tif')
img = cv2.imread(inpFile)

#covert image to grayscale and then extract edges
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)

#Hough Line parameters
minLineLength = 10
maxLineGap = 2

#Creating line segments out of Image
lines = cv2.HoughLinesP(gray,.75,np.pi/180,50,minLineLength,maxLineGap)

myData = [['ImageId', 'WTxtd']]
i = 1;

#Initialize point vector
points = [[0,1,[0]]]
for x1,y1,x2,y2 in lines[0]:
	addPoint1 = True
	addPoint2 = True
	#Check to see if a similar point is already in the vector
	for checking in points:
		if(abs(x1-checking[0])+abs(y1-checking[1]) < 20):
			addPoint1 = False
			connection1 = points.index(checking)
	for checking in points:
		if(abs(x2-checking[0])+abs(y2-checking[1]) < 20):
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
cv2.imwrite('MaybeThisTime.jpg',img)

#Create CSV file
myFile = open(outFile,'w')
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
