# Spacenet-round3

### The SpaceNet Challenge Round 3
In this challenge, we were tasked with finding automated methods for extracting map-ready road networks from high-resolution satellite imagery. Moving towards more accurate fully automated extraction of road networks will help bring innovation to computer vision methodologies applied to high-resolution satellite imagery, and ultimately help create better maps where they are needed most.
The goal is to extract navigable road networks that represent roads from satellite images. The linestrings the algorithm returns is compared to ground truth data, and the quality of the solution is judged by the Average Path Length Similarity (APLS) metric.

* The official problem statement: https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17036&pm=14735. 
* The challenge website: https://www.topcoder.com/spacenet. 
* How the scoring function works: https://medium.com/the-downlinq/spacenet-road-detection-and-routing-challenge-part-i-d4f59d55bfce.
* Data: https://aws.amazon.com/public-datasets/spacenet/

We solved the road detection problem in SpacenNet challenge as a semantic segmentation task in computer vision. Our model is based on a variant of fully convolutional neural network, U-Net [https://arxiv.org/abs/1505.04597]. U-Net is one of the most successful and popular convolutional neural network architecture for image segmentation.
 
 
### TO DO: More details about the codes and the algorithm will be added soon... 
<!--  
### Solution
##### Input
##### Segmentation
* get code on github from maskrcnn
* discuss how it works
* decide if best way to start
### Notes
chmod g+rw [fileToModify.txt]
chmod -r g+rw *
 -->
##### Linestring Conversion
First, we created a function which would draw the ground truth LineString objects into a binary mask, with different width settings available for the road.

<img width="400" alt="screen shot 2018-01-26 at 10 20 49 am" src="https://user-images.githubusercontent.com/6694735/35446429-abdc48fc-0282-11e8-9ef7-2fa47774d58d.png">

<img width="400" alt="screen shot 2018-01-26 at 10 29 19 am" src="https://user-images.githubusercontent.com/6694735/35447016-819366a0-0284-11e8-9d61-85e203aa9a66.png">

<!-- Then, we took on converting an image mask (which will be generated by our network) and writing the corresponding LineString objects. Our first attempt used edge detection on the mask, and then hough line transformation to generate lines. We quickly dropped the edge detection part, since our images were already binary. After spending time altering the settings and developing a cool solution to merge close endpoints, we decided the result was not good enough, and went back to basics. -->

Here are the results of skeletonizing a 10 pixel wide mask.

<img width="400" alt="screen shot 2018-01-26 at 10 35 18 am" src="https://user-images.githubusercontent.com/6694735/35447081-b41bd71a-0284-11e8-9476-a9ef589ef845.png">

Assuming we could generate single lines roads (possible with the skeletonizing function), we would follow a line of white pixels until that line ended, writing each point into the linestring. At an intersection, other paths would be saved to explore later. For our test example, we were able to perfectly replicate the image, using 16 linestrings but 7000 points. We further improved this solution to reduce the number of points to 3500 by tracking our current direction, and only adding a point when the direction changed.

Here is the mask generated when we draw the linestrings extracted from the skeletonized 10 pixel image.

<img width="400" alt="screen shot 2018-01-26 at 10 35 35 am" src="https://user-images.githubusercontent.com/6694735/35447089-b6f35904-0284-11e8-98b8-8760fddd325f.png">

<!--
### Contributors
* Ramin Nabati
* Alireza Rahimpour
* Elliot Greenlee
* Razieh Kaviani
* Steven Patrick
-->
