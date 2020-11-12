#import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import picamera.array


#global colour1Confidence
#global colour1Target
#colour1Confidence = [0,0,0,0,0]
#colour1Target = [0,0,0,0,0]
#colour1Good = False

global black1Confidence
global gradBlack
black1Confidence = [0,0,0,0,0]
gradBlack = [0,0,0,0,0]
red1Good = False

global black2Confidence
global intblack
black2Confidence = [0,0,0,0,0]
intRed = [0,0,0,0,0]
red2Good = False

global black1lConfidence
global gradlBlack
black1lConfidence = [0,0,0,0,0]
gradlBlack = [0,0,0,0,0]
black1lGood = False

global black2lConfidence
global intlBlack
black2lConfidence = [0,0,0,0,0]
intlBlack = [0,0,0,0,0]
black2lGood = False

blueland = False

global imagescale
imagescale = 35

global framecount
framecount = 0

global M
M = np.array ([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], dtype = "float32")

# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
perspx = 120 # Don't change this as it changes the height of the warped image.
pts = np.array([(0+perspx, 0), (319-perspx, 0),(319,239),(0, 239) ] , dtype = "float32")
         
# Order Points
# initialzie a list of coordinates that will be ordered
# such that the first entry in the list is the top-left,
# the second entry is the top-right, the third is the
# bottom-right, and the fourth is the bottom-left
rect = np.zeros((4, 2), dtype = "float32")
#rect = pts

# the top-left point will have the smallest sum, whereas
# the bottom-right point will have the largest sum
s = pts.sum(axis = 1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

# now, compute the difference between the points, the
# top-right point will have the smallest difference,
# whereas the bottom-left will have the largest difference
diff = np.diff(pts, axis = 1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

# Do 4 Point transform
(tl, tr, br, bl) = rect

# compute the width of the new image, which will be the
# maximum distance between bottom-right and bottom-left
# x-coordiates or the top-right and top-left x-coordinates
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

# compute the height of the new image, which will be the
# maximum distance between the top-right and bottom-right
# y-coordinates or the top-left and bottom-left y-coordinates
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

# now that we have the dimensions of the new image, construct
# the set of destination points to obtain a "birds eye view",
# (i.e. top-down view) of the image, again specifying points
# in the top-left, top-right, bottom-right, and bottom-left
# order

dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

# compute the perspective transform matrix and then apply it
M = cv2.getPerspectiveTransform(rect, dst)
#print M

def preparemask (hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper);
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask;

def meldmask (mask_0, mask_1):
    mask = cv2.bitwise_or(mask_0, mask_1)
    return mask;


class PiVideoStream:

        global M
        
         
	def __init__(self, resolution=(320, 240), framerate=32):
               
		# initialize the camera and stream
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.camera.rotation =-90
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)

		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.frame = None
		self.stopped = False


	def start(self):

                global M
                
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		print("Thread starting")
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		for f in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
			self.frame = f.array
			self.rawCapture.truncate(0)

			# if the thread indicator variable is set, stop the thread
			# and resource camera resources
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return
	def readfollow(self):

                global black1lConfidence
                global gradBlack
                global black2Confidence
                global intBlack
                global imagescale
                global M
                global framecount
                global circle
                global rlcontrol
                global bluex
                global bluey
                global moving_ratio
                global blueland
                
                #framecount = framecount + 1
                #if framecount == 100:
                #    time.sleep(0.5)
                #    framecount  = 0

                # Set the image resolution.
                xres = 320
                yres = 240
                xColour1 = xRed = 0.0

                # Initialise confidence to indicate the line has not been located with the current frame
                newblack1lConfidence = 0
                black1Good = False
                newBlack2Confidence = 0
                black2Good = False

                # Initialise variables for line calculations
                xBlack1 = xBlack2 = 0
                yBlack1 = yBlack2 = 0
                intc = m = 0.0
                dx = dy = 0
                bearing = offset = circle = 0
                cx_green = cy_green = 0
                cx_black1 = cx_black2 = 0
                
                # return the frame most recently read
                frame = self.frame

                # apply the four point tranform to obtain a "birds eye view" of
                # the image

                warped = frame
                #warped = cv2.warpPerspective(frame, M, (320+(8*imagescale), 268))
                height = 240
                width = 320
                #channels 

                # Set y coords of regions of interest.
                # The upper and lower bounds
                roidepth = 20    # vertical depth of regions of interest
                roiymin = 40    # minumum ranging y value for roi origin
                roiymintop = roiymin - roidepth
                roiymax = height - roidepth -1   # maximum ranging y value for bottom roi origin
                
                # Convert to hsv and define region of interest before further processing.
                fullhsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                
                # green = 60
                # blue = 120;
                # yellow = 30;
                #Colour1 = 60

                # Set the sensitivity of the hue
                #sensitivity = 20

                # Black is a special case as it sits either side of 0 on the HSV spectrum
                # So we create two masks, one greater than zero and one less than zero
                # Then combine the two.
                lower_black_0 = np.array([0, 0, 0])
                upper_black_0 = np.array([179, 255, 10])
                
                lower_black_1 = np.array([0, 0, 0])
                upper_black_1 = np.array([179, 255, 63])

                
                # Initialise the bottom roi at the maximum limit
                y3 = roiymax
                y4 = y3 + roidepth

                while y3 > roiymin:

                    # This defines the lower band, looking closer in
                    roihsv2 = fullhsv[y3:y4, 0:(width-1)]

                    # Prepare the masks for the lower roi 
                    maskr_2 = preparemask (roihsv2, lower_black_0 , upper_black_0)
                    maskr_3 = preparemask (roihsv2, lower_black_1 , upper_black_1 )
                    maskr2 = meldmask ( maskr_2, maskr_3)
            
                    # find contours in the lower roi and initialize the center
                    cnts_black2 = cv2.findContours(maskr2.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
                    center2 = None

                    # Now to find the tracking line in the lower roi
                    # only proceed if at least one contour was found
                    if len(cnts_black2) > 0:
                        
                        # find the largest contour in the mask, then use
                        # it to compute the minimum enclosing circle and
                        # centroid
                        c_black2 = max(cnts_black2, key=cv2.contourArea)
                        ((x_black2, y_black2), radius_black2) = cv2.minEnclosingCircle(c_black2)
                        M_black2 = cv2.moments(c_black2)

                        # compute the center of the contour
                        cx_black2 = int(M_black2["m10"] / M_black2["m00"])
                        cy_black2 = int(M_black2["m01"] / M_black2["m00"])
                        

                        # cy_red is set in the region of interest, so need to adjust for origin in frame
                        cy_black2 = cy_black2 + y3
                        # center = ( cx_red, cy_red )

                        # only proceed if the radius meets a minimum size
                        if radius_black2 > 5:
                            newBlack2Confidence = 100
                            # draw the circle and centroid on the frame
                            #cv2.circle(warped, (cx_red2, cy_red2), int(radius_red2),
                            #(0, 0, 255), 2)
                            # cv2.circle(frame, center, 5, (0, 0, 255), -1)

                            # calculate offset
                            xBlack2 = cx_black2 - (width/2) # Contrived so pstve to right of centreline
                            yBlack2 = height - cy_black2  # Adjust to make origin bottom centre of image

                            # The target has been found, so we can break out of the loop here
                            break

                    # But here the target has not been found, we need to move the ROI up
                    y3 = y3 - roidepth
                    y4 = y3 + roidepth

                # And here we have either hit the buffers or found the target.

                
                # So now try for the top roi, working down.                      
                # Initialise the top roi at the very top
                y1 = 0
                y2 = y1 + roidepth

                while y2 < y3: # Go as far as the lower roi but no more.

                    newblack1lConfidence = 0

                    # This defines the upper roi, looking further away
                    roihsv1 = fullhsv[y1:y2, 0:(width-1)]
                    #print(roihsv1)

                    # Prepare the masks for the top roi 
                    maskr_0 = preparemask (roihsv1, lower_black_0 , upper_black_0)
                    maskr_1 = preparemask (roihsv1, lower_black_1 , upper_black_1 )
                    maskr1 = meldmask ( maskr_0, maskr_1)

                    # find contours in the upper roi and initialize the center
                    cnts_black1 = cv2.findContours(maskr1.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
                    center1 = None

                    # Now to find the tracking line in the upper roi
                    # only proceed if at least one contour was found
                    if len(cnts_black1) > 0:
                        #print len(cnts_red1)
                        # find the largest contour in the mask, then use
                        # it to compute the minimum enclosing circle and
                        # centroid
                        c_black1 = max(cnts_black1, key=cv2.contourArea)
                        ((x_black1, y_black1), radius_black1) = cv2.minEnclosingCircle(c_black1)
                        M_black1 = cv2.moments(c_black1)

                        # compute the center of the contour
                        cx_black1 = int(M_black1["m10"] / M_black1["m00"])
                        cy_black1 = int(M_black1["m01"] / M_black1["m00"])
                        

                        # cy_red is set in the region of interest, so need to adjust for origin in frame
                        cy_black1 = cy_black1 + y1
                        # center = ( cx_red, cy_red )

                        # only proceed if the radius meets a minimum size
                        if radius_black1 > 5:
                            newblack1lConfidence = 100
                            # draw the circle and centroid on the frame
                            #cv2.circle(warped, (cx_red1, cy_red1), int(radius_red1),
                            #(0, 0, 255), 2)
                            # cv2.circle(frame, center, 5, (0, 0, 255), -1)

                            # calculate offset
                            xBlack1 = cx_black1-(width/2)   # Contrived so pstve to right of centreline
                            yBlack1 = height - cy_black1  # Adjust to make origin bottom centre of image
                            

                            # The target has been found, so we can break out of the loop here
                            break

                    # But here the target has not been found, we need to move the ROI down
                    y1 = y1 + roidepth
                    y2 = y1 + roidepth

                # And here we have either hit the buffers or found the target.


                if (newblack1lConfidence == 100 and newBlack2Confidence == 100):
                    # Calculate gradient and intercept for thes valid pair of points.
                    # The aspect ratio of warped image is wrong (should be long and thin!).
                    #Rather than stretch the image (processing time), we simply apply a multiple to the
                    #y axis.
                    aspect = 1.0
                
                    dy = yBlack1 - yBlack2
                    dx = xBlack1 - xBlack2
                    m = float(dx)/(aspect*float(dy))
                    intc = xBlack2 - (m * float(yBlack2))

                                         
                    # Add to the running average for each.
                    # Update gradient array and calculate running average to return as target gradient
                    # 
                    gradBlack[4] = gradBlack[3]
                    gradBlack[3] = gradBlack[2]
                    gradBlack[2] = gradBlack[1]
                    gradBlack[1] = gradBlack[0]
                    gradBlack[0] = m
                    # Update the gradient for the bearing signal from the last 5 on a running average
                    m = (gradBlack[0]+gradBlack[1]+gradBlack[2]+gradBlack[3]+gradBlack[4])/5


                    # Update intercept array and calculate running average to return as target intercept
                    intBlack[4] = intBlack[3]
                    intBlack[3] = intBlack[2]
                    intBlack[2] = intBlack[1]
                    intBlack[1] = intBlack[0]
                    intBlack[0] = intc
                    # Update the x axis intercept for the offset signal from the last 5 on a running average
                    intc = (intBlack[0]+intBlack[1]+intBlack[2]+intBlack[3]+intBlack[4])/5


                # The confidence running averages are updated whether the lock was successful or not
                # Update confidence array for lower roi
                black1lConfidence[4] = black1lConfidence[3]
                black1lConfidence[3] = black1lConfidence[2]
                black1lConfidence[2] = black1lConfidence[1]
                black1lConfidence[1] = black1lConfidence[0]
                black1lConfidence[0] = newblack1lConfidence
                newblack1lConfidence = (black1lConfidence[0]+black1lConfidence[1]+black1lConfidence[2]+black1lConfidence[3]+black1lConfidence[4])/5


                # Update confidence array for upper roi
                black2Confidence[4] = black2Confidence[3]
                black2Confidence[3] = black2Confidence[2]
                black2Confidence[2] = black2Confidence[1]
                black2Confidence[1] = black2Confidence[0]
                black2Confidence[0] = newBlack2Confidence
                newBlack2Confidence = (black2Confidence[0]+black2Confidence[1]+black2Confidence[2]+black2Confidence[3]+black2Confidence[4])/5


                # In following mode, we must have lock on both rois.  So red1Good and red2Good = True
                # Now to calculate signals to be returned, normalised between -1 and 1.
                if (newblack1lConfidence > 30) and (newBlack2Confidence > 30):
                    black1Good = black2Good = True
                    offset = intc / (width/2) # This gives use the abiity to respond to values off the camera at y=0 !(so beyond 1)     
                    bearing = np.degrees(np.arctan(m)) # To move towards the target.    Bearing is in degrees.
                    imagescale = int(np.absolute(bearing)) # This is used to modulate pitch, so must always be positive
                    if imagescale > 35:
                        imagescale = 35
                else:
                    imagescale = 35 # We have lost lock, so this sets the image width to maximum .

                # Draw Region of interest
                '''
                cv2.line(warped, (0, y1), (width, y1), (255,0,0))
                cv2.line(warped, (0, y2), (width, y2), (255,0,0))
                cv2.line(warped, (0, y3), (width, y3), (255,0,0))
                cv2.line(warped, (0, y4), (width, y4), (255,0,0))
                
                '''
                # print "Following", " Bearing: ",bearing, red1Good, "  Offset: ", offset, red2Good
                #------------------------green--------------------------------------#
               
                roihsv3 = fullhsv[70:170, 110:210]

                lower_blue_0 = np.array([0, 0, 0])
                upper_blue_0 = np.array([100, 150, 0])

                lower_blue_1 = np.array([0, 0, 0])
                upper_blue_1 = np.array([140, 255, 255])

                # Prepare the masks for the lower roi 
                maskr_33 = preparemask (roihsv3, lower_blue_0 , upper_blue_0)
                maskr_44 = preparemask (roihsv3, lower_blue_1 , upper_blue_1)
                maskr3= meldmask ( maskr_33, maskr_44)

                # find contours in the lower roi and initialize the center
                cnts_green = cv2.findContours(maskr3.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[-2]

                if maskr3 is not None:
                    cnts_blue1 = cv2.findContours(maskr3.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
                    centerb1 = None
                    if len(cnts_blue1) > 0:
                        
                        # find the largest contour in the mask, then use
                        # it to compute the minimum enclosing circle and
                        # centroid
                        c_blue1 = max(cnts_blue1, key=cv2.contourArea)
                        ((x_blue1, y_blue1), radius_blue1) = cv2.minEnclosingCircle(c_blue1)
                        M_blue1 = cv2.moments(c_blue1)

                        # compute the center of the contour
                        cx_blue1 = int(M_blue1["m10"] / M_blue1["m00"])
                        cy_blue1 = int(M_blue1["m01"] / M_blue1["m00"])
                        bluex = cx_blue1
                        bluey = cy_blue1
                        blueland = True

                moving_ratio = 0
                if cx_black1 <= 0 and cx_black2 <= 0:
                    rlcontrol = 0
                else:
                    if cx_black1 > 190 and cx_black2 > 190:
                        rlcontrol=1#aircraft should go left
                        #moving_ratio = ((cx_red1 + cx_red2)/2 - 190)/1.7
                        #print("distance: %.2f moving_ratio: %.2f"%(moving_ratio*1.7,moving_ratio))
                        
                    elif cx_black1 < 130 and cx_black2 < 130:
                        rlcontrol=2#aircraft should go right
                        #moving_ratio = (130 - (cx_red1 + cx_red2)/2)/1.7
                        #print("distance: %.2f moving_ratio: %.2f"%(moving_ratio*1.7,moving_ratio)) 
                        
                    else:
                        rlcontrol=3#aircraft do not move
                        
                #if cx_green !=0 and cy_green!=0:
                #    cv2.rectangle(warped, (cx_green-5,cy_green-5), (cx_green+5,cy_green+5), (0,128,255),-1)
                #cv2.imshow("Warped", warped)
                key = cv2.waitKey(1) & 0xFF
		return (bearing, black1Good, offset, black2Good, circle, rlcontrol, moving_ratio, blueland)

	def readlost(self):


                global black1lConfidence
                global gradBlack
                global black2Confidence
                global intBlack
                global bearing
                global M
                global circle
                global bluex
                global bluey
                global moving_ratio
                global blueland


                # Set the image resolution.
                xres = 320
                yres = 240
                xColour1 = xRed = 0.0
                

                # Initialise confidence to indicate the line has not been located with the current frame
                newblack1lConfidence = 0
                blackl1Good = False
                newblack2lConfidence = 0
                blackl2Good = False

                # Initialise variables for line calculations
                xBlack1 = xBlack2 = 0
                yBlack1 = yBlack2 = 0
                intc = m = 0.0
                dx = dy = 0
                bearing = offset = 0.0
                circle = 0
                cx_Black1 = 0
                # return the frame most recently read
                frame = self.frame

                # apply the four point tranform to obtain a "birds eye view" of
                # the image.  We are mapping this onto an extended image to get the broadest horizon.

                ewarped = frame
	        #ewarped = cv2.warpPerspective(frame, M, (600, 268))

                height = 240
                width = 320

                # Set y coords of regions of interest.
                # The upper and lower bounds
                roidepth = 20    # vertical depth of regions of interest
                roiymin = 40    # minumum ranging y value for roi origin
                roiymintop = roiymin - roidepth
                roiymax = height - roidepth -1   # maximum ranging y value for bottom roi origin

                
                # Convert to hsv and define region of interest before further processing.
                fullhsv = cv2.cvtColor(ewarped, cv2.COLOR_BGR2HSV)

                #green = 60
                #blue = 120;
                #yellow = 30;
                #Colour1 = 60
                # Set the sensitivity of the hue
                #sensitivity = 20
                # Red is a special case as it sits either side of 0 on the HSV spectrum
                # So we create two masks, one greater than zero and one less than zero
                # Then combine the two.
                
                lower_black_0 = np.array([0,   0,   0])
                upper_black_0 = np.array([179, 255, 10])
                
                lower_black_1 = np.array([0,  0,   0])
                upper_black_1 = np.array([179,255, 63])


                # Initialise the bottom roi at the maximum limit
                y3 = roiymax
                y4 = y3 + roidepth
                xBlack2 = yBlack2 = 0 # Anchor the first point at the origin.  We will seek the target with the upper roi.
                """
                while y3 > roiymin:
                    # This defines the lower band, looking closer in
                    roihsv2 = fullhsv[y3:y4, 0:(width-1)]
                    # Prepare the masks for the lower roi 
                    maskr_2 = preparemask (roihsv2, lower_red_0 , upper_red_0)
                    maskr_3 = preparemask (roihsv2, lower_red_1 , upper_red_1 )
                    maskr2 = meldmask ( maskr_2, maskr_3)
                    # find contours in the lower roi and initialize the center
                    cnts_red2 = cv2.findContours(maskr2.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
                    center2 = None
                    # Now to find the tracking line in the lower roi
                    # only proceed if at least one contour was found
                    if len(cnts_red2) > 0:
                        
                        # find the largest contour in the mask, then use
                        # it to compute the minimum enclosing circle and
                        # centroid
                        c_red2 = max(cnts_red2, key=cv2.contourArea)
                        ((x_red2, y_red2), radius_red2) = cv2.minEnclosingCircle(c_red2)
                        M_red2 = cv2.moments(c_red2)
                        # compute the center of the contour
                        cx_red2 = int(M_red2["m10"] / M_red2["m00"])
                        cy_red2 = int(M_red2["m01"] / M_red2["m00"])
                        
                        # cy_red is set in the region of interest, so need to adjust for origin in frame
                        cy_red2 = cy_red2 + y3
                        # center = ( cx_red, cy_red )
                        # only proceed if the radius meets a minimum size
                        if radius_red2 > 5:
                            newred2lConfidence = 100
                            # draw the circle and centroid on the frame
                            cv2.circle(ewarped, (cx_red2, cy_red2), int(radius_red2),
                            (0, 0, 255), 2)
                            # cv2.circle(frame, center, 5, (0, 0, 255), -1)
                            # calculate offset
                            xRed2 = cx_red2 - (width/2) # Contrived so pstve to right of centreline
                            yRed2 = height - cy_red2  # Adjust to make origin bottom centre of image
                            # The target has been found, so we can break out of the loop here
                            break
                    # But here the target has not been found, we need to move the ROI up
                    y3 = y3 - roidepth
                    y4 = y3 + roidepth
                # And here we have either hit the buffers or found the target.
                """
                
                # So now try for the top roi, working down.                      
                # Initialise the top roi at the very top
                y1 = 0
                y2 = y1 + roidepth

                while y2 < y3: # Go as far as the lower roi but no more.

                    newblack1lConfidence = 0

                    # This defines the upper roi, looking further away
                    roihsv1 = fullhsv[y1:y2, 0:(width-1)]                 

                    # Prepare the masks for the top roi 
                    maskr_0 = preparemask (roihsv1, lower_black_0 , upper_black_0)
                    maskr_1 = preparemask (roihsv1, lower_black_1 , upper_black_1 )
                    maskr1 = meldmask ( maskr_0, maskr_1)

                    # find contours in the upper roi and initialize the center
                    cnts_black1 = cv2.findContours(maskr1.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
                    center1 = None

                    # Now to find the tracking line in the upper roi
                    # only proceed if at least one contour was found
                    if len(cnts_black1) > 0:
                        
                        # find the largest contour in the mask, then use
                        # it to compute the minimum enclosing circle and
                        # centroid
                        c_black1 = max(cnts_black1, key=cv2.contourArea)
                        ((x_black1, y_black1), radius_black1) = cv2.minEnclosingCircle(c_black1)
                        M_black1 = cv2.moments(c_black1)

                        # compute the center of the contour
                        cx_Black1 = int(M_black1["m10"] / M_black1["m00"])
                        cy_black1 = int(M_black1["m01"] / M_black1["m00"])
                        

                        # cy_red is set in the region of interest, so need to adjust for origin in frame
                        cy_black1 = cy_black1 + y1
                        # center = ( cx_red, cy_red )

                        # only proceed if the radius meets a minimum size
                        if radius_black1 > 5:
                            newblack1lConfidence = 100
                            # draw the circle and centroid on the frame
                            #cv2.circle(ewarped, (cx_red1, cy_red1), int(radius_red1),
                            #(0, 0, 255), 2)
                            # cv2.circle(frame, center, 5, (0, 0, 255), -1)

                            # calculate offset
                            xBlack1 = cx_Black1-(width/3)   # Contrived so pstve to right of centreline
                            yBlack1 = height - cy_black1  # Adjust to make origin bottom centre of image

                            # The target has been found, so we can break out of the loop here
                            break

                    # But here the target has not been found, we need to move the ROI down
                    y1 = y1 + roidepth
                    y2 = y1 + roidepth

                # And here we have either hit the buffers or found the target.


                if newblack1lConfidence == 100 : # So the bottom end is already set and anchored at the origin.
                    # Calculate gradient and intercept for thes valid pair of points.
                    # The aspect ratio of warped image is wrong (should be long and thin!).
                    # Rather than stretch the image (processing time), we simply apply a multiple to the
                    # y axis.
                    aspect = 1.0
                
                    dy = yBlack1 - yBlack2
                    dx = xBlack1 - xBlack2
                    m = float(dx)/(float(dy))
                    intc = 0 # This is always the case because the lower roi target is locked to the origin
                                         
                    # Add to the running average for each.
                    # Update gradient array and calculate running average to return as target gradient
                    # 
                    gradBlack[4] = gradBlack[3]
                    gradBlack[3] = gradBlack[2]
                    gradBlack[2] = gradBlack[1]
                    gradBlack[1] = gradBlack[0]
                    gradBlack[0] = m
                    # Update the gradient for the bearing signal from the last 5 on a running average
                    m = (gradBlack[0] + gradBlack[1] + gradBlack[2] + gradBlack[3] + gradBlack[4]) / 5

                    # Update intercept array and calculate running average to return as target intercept
                    intBlack[4] = intBlack[3]
                    intBlack[3] = intBlack[2]
                    intBlack[2] = intBlack[1]
                    intBlack[1] = intBlack[0]
                    intBlack[0] = intc
                    # Update the x axis intercept for the offset signal from the last 5 on a running average
                    intc = (intBlack[0] + intBlack[1] + intBlack[2] + intBlack[3] + intBlack[4]) / 5

                    # The confidence running averages are updated whether the lock was successful or not
                    # Update confidence array for lower roi
                    black1lConfidence[4] = black1lConfidence[3]
                    black1lConfidence[3] = black1lConfidence[2]
                    black1lConfidence[2] = black1lConfidence[1]
                    black1lConfidence[1] = black1lConfidence[0]
                    black1lConfidence[0] = newblack1lConfidence
                    newblack1lConfidence = (black1lConfidence[0] + black1lConfidence[1] + black1lConfidence[2] +
                                            black1lConfidence[3] + black1lConfidence[4]) / 5

                    # Update confidence array for upper roi
                    black2Confidence[4] = black2Confidence[3]
                    black2Confidence[3] = black2Confidence[2]
                    black2Confidence[2] = black2Confidence[1]
                    black2Confidence[1] = black2Confidence[0]
                    black2Confidence[0] = newBlack2Confidence
                    newBlack2Confidence = (black2Confidence[0] + black2Confidence[1] + black2Confidence[2] +
                                           black2Confidence[3] + black2Confidence[4]) / 5

                # Now to calculate signals to be returned, normalised between -1 and 1.
                if newblack1lConfidence > 30:
                    blackl1Good = True
                    offset = intc # Recall intc locked to zero, so no offset here.
                    bearing = np.degrees(np.arctan(m)) # To move towards the target.  Bearing is in degrees.


                # Draw Region of interest
                #cv2.line(ewarped, (0, y1), (width, y1), (255,0,0))
                #cv2.line(ewarped, (0, y2), (width, y2), (255,0,0))
                #cv2.line(ewarped, (0, y3), (width, y3), (255,0,0))
                #cv2.line(ewarped, (0, y4), (width, y4), (255,0,0))

                # print "Lost", bearing, red1Good, offset, red2Good

                #cv2.imshow('Frame',frame)
                # cv2.imshow("Extended Warped", ewarped)
                roihsv3 = fullhsv[70:170, 110:210]

                lower_blue_0 = np.array([0, 0, 0])
                upper_blue_0 = np.array([100, 150, 0])

                lower_blue_1 = np.array([0, 0, 0])
                upper_blue_1 = np.array([140, 255, 255])

                # Prepare the masks for the lower roi
                maskr_33 = preparemask(roihsv3, lower_blue_0, upper_blue_0)
                maskr_44 = preparemask(roihsv3, lower_blue_1, upper_blue_1)
                maskr3 = meldmask(maskr_33, maskr_44)

                cnts_blue2 = cv2.findContours(maskr3.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
                centerb2 = None

                # Now to find the tracking line in the upper roi
                # only proceed if at least one contour was found
                if len(cnts_blue2) > 0:

                    # find the largest contour in the mask, then use
                    # it to compute the minimum enclosing circle and
                    # centroid
                    c_blue2 = max(cnts_blue2, key=cv2.contourArea)
                    ((x_blue2, y_blue2), radius_blue2) = cv2.minEnclosingCircle(c_blue2)
                    M_blue2 = cv2.moments(c_blue2)

                    # compute the center of the contour
                    cx_blue2 = int(M_blue2["m10"] / M_blue2["m00"])
                    cy_blue2 = int(M_blue2["m01"] / M_blue2["m00"])
                    bluex = cx_blue2
                    bluey = cy_blue2
                    blueland = True

                moving_ratio = 0
                if cx_Black1 <= 0:
                    rlcontrol = 0
                else:
                    if cx_Black1 > 190:
                        rlcontrol=1#aircraft should go left
                        #moving_ratio = (cx_red1 - 190)/1.7
                        #print("distance: %.2f moving_ratio: %.2f"%(moving_ratio*1.7,moving_ratio))
                        
                    elif cx_Black1 < 130:
                        rlcontrol=2#aircraft should go right
                        #moving_ratio = (130 - cx_red1)/1.7
                        #print("distance: %.2f moving_ratio: %.2f"%(moving_ratio*1.7,moving_ratio))
                        
                    else:
                        rlcontrol=3#aircraft do not move
                        
                #cv2.imwrite('/home/pi/images/image'+ (time.strftime("%H:%M:%S"))+'.jpg', frame)
                key = cv2.waitKey(1) & 0xFF
                #cv2.imshow("Warped", ewarped)
		return (bearing, blackl1Good, offset, blackl2Good, circle, rlcontrol, moving_ratio, blueland)

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
