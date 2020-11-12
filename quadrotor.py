# import the necessary packages
# from __future__ import print_function
# This needs to be on the first line
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil # Needed for command message definitions
import time
import numpy as np
from PiVideoStream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import cv2
import sys
import picamera
import picamera.array
from threading import Thread
import sys
import RPi.GPIO as GPIO
import math

#--------------------------SET UP VIDEO THREAD ----------------------------------
print('sampling THREADED frames from picamera')
vs = PiVideoStream().start()
#--------------------------FUNCTION DEFINITION FOR SET_ATTITUDE MESSAGE -------
def set_attitude (pitch, roll, yaw):
    print"set_attitude_yaw"
    print("yaw_o: %d"%yaw)
    
    #yaw = np.radians(yaw) 
    yaw = yaw*10
    yaw = round(yaw)
    if yaw > 0:
        yaw=1500+yaw
    elif yaw < 0:
        yaw=1500+yaw
    else:
        yaw=1500
    print("yaw_c: %d"%yaw)
    
    vehicle.channels.overrides = {'1':1480,'2':1492,'3':1500,'4':yaw}
    time.sleep(0.5)
    vehicle.flush()
    print("ch1:%d ch2:%d ch3:%d ch4:%d"%(vehicle.channels['1'],vehicle.channels['2'],vehicle.channels['3'],vehicle.channels['4']))
#----------------------------------------------------------------#
def set_attitude1(pitch, roll):
    print"set_attitude_roll"
    if roll > 0:
        roll=1500+roll   
    elif roll < 0:
        roll=1500+roll
    else:
        roll=1500
    print("roll: %d "%roll)
    vehicle.channels.overrides = {'1':roll,'2':1488,'3':1500}
    time.sleep(0.4)
    vehicle.channels.overrides = {'1':1500,'2':1492,'3':1500}
    time.sleep(0.1)
    vehicle.flush()
    print("ch1:%d ch2:%d ch3:%d"%(vehicle.channels['1'],vehicle.channels['2'],vehicle.channels['3']))

#---------------Exception to land--------------------------------------------------------
def land():
    print("start landing")
    thro = vehicle.channels['3']
    while True:
        thro -= 15
        time.sleep(1)
        vehicle.channels.overrides['3']=thro
        vehicle.flush()
        print vehicle.channels['3']
        if thro < 900:
            break
    print "Close vehicle object"
    vehicle.close()
    cv2.destroyAllWindows()
    vs.stop()
    sys.exit()
#----------------------------------------------------------#
def servo():
    pwm = GPIO.PWM(11,50)
    pwm.start(2.3)
    time.sleep(0.1)
    #pwm.start(12.3)
    #time.sleep(1)
#-------------- FUNCTION DEFINITION TO ARM AND TAKE OFF TO GIVEN ALTITUDE ---------------
def arm_and_takeoff(aTargetAltitude):
    if vehicle.channels['5']>1700:
        land()   
    thro = 1400
    print "@@@ Taking off @@@"
    while True:
        print 'Distance: %.2f'%(vehicle.rangefinder.distance)    
        if vehicle.channels['5']>1700:
            land()
        #Break and return from function just below target altitude.        
        if vehicle.rangefinder.distance>=aTargetAltitude*0.88: 
            print "Reached target altitude"
            vehicle.mode=VehicleMode("ALT_HOLD")
            vehicle.channels.overrides = {'2':1500,'3':1500}
            time.sleep(0.1)
            vehicle.flush()
            break
        else:
            if thro < 1500:
                thro+=100
            elif thro < 1590:
                thro+=50
                if thro > 1590:
                    thro = 1600
            else:
                thro=1610
            vehicle.channels.overrides = {'1':1468,'2':1517,'3':thro}
            time.sleep(0.1)
            print vehicle.channels['3']
#-----------------------------------------------------------------------------------------
#------------------------------TRACKING Program--------------------------------------------
#-------------- FUNCTION DEFINITION TO FLY IN VEHICLE STATE TRACKING  ---------------------
def tracking (vstate):
    print vstate
    
    red1Good = red2Good = False # Set True when returned target offset is reliable.
    bearing = offset = 0
    target = None
    
    if vehicle.channels['5']>1700:
        land()
        
    while vstate == "tracking":
     
        # grab the frame from the threaded video stream and return left line offset
        # We do this to know if we have a 'lock' (goodTarget) as we come off of manual control.
        target = vs.readfollow()
        bearing = target[0]
        red1Good = target[1]
        offset = target[2]
        red2Good = target[3]
        circle = target[4]
        rlcontol = target[5]
        
        if vehicle.channels['5']>1700:
            land()
        
        # Check if operator has transferred to autopilot using TX switch.
        if vehicle.mode == "ALT_HOLD":
            # print "In Guided mode..."
            if ( red1Good and red2Good ) == True:
                # print "In guided mode, setting following..."
                vstate = "following"
            else:
                # print "In guided mode, setting lost..."
                vstate = "lost"
                
    return vstate
#-----------------------------------------------------------------------------------------
#------------------------------FOLLOWING Program------------------------------------------
#-------------- FUNCTION DEFINITION TO FLY IN VEHICLE STATE FOLLOWING---------------------
def following (vstate):
    #print vstate+" mode"
    print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
    global aTargetAltitude   
    #The vehicle process images and uses all data to fly in the following state.
    # It sends attitude messages until manual control is resumed.    
    maxPitch = -1   # Maximum pitch for target at 0 bearing.
    multRoll =  8     # The roll angle if offset is at edge of the near field of view.
    multYaw  =  0     # A multiplier for the rate of Yaw.  
    yaw = roll = 0  # Initialise
    target = None   # Initialise tuple returned from video stream
       
    while vstate =="following":
        target = vs.readfollow()
        bearing = target[0] # Returned in degrees, +ve clockwise
        red1Good = target[1]
        offset = target[2]    # Returned as a fraction of image width.  -1 extreme left.
        red2Good = target[3]
        circle = target[4]
        rlcontrol = target[5]
        moving_ratio = target[6]
        blueland = target[7]
        if vehicle.channels['5']>1700:
                land()

        if blueland == True:
            land()

        if rlcontrol == 1:
            pitch1=-1
            #roll1 = moving_ratio*0.3
            roll1 = 25
            print ('Following mode rlcontrol===> %d'%rlcontrol)
            set_attitude1(pitch1,roll1)

        
        elif rlcontrol == 2:
            pitch1=-1
            #roll1 = -moving_ratio*0.3
            roll1 = -35
            print ('Following mode rlcontrol===> %d'%rlcontrol)
            set_attitude1(pitch1,roll1)

        
        elif rlcontrol == 3:
            pitch1=-1
            roll1=0
            print ('Following mode rlcontrol===> %d'%rlcontrol)
            set_attitude1(pitch1,roll1)
        else:
            print"No Line"# cx_red1 & cx_red2 <=0
            
        # grab the frame from the threaded video stream and return left line offset
        # We do this to know if we have a 'lock' (goodTarget) as we come off of manual control.

        #print("Distance: %.3f"%vehicle.rangefinder.distance)
        #print("bearing:  %.2f"%bearing)

        # Check if operator has transferred to autopilot using TX switch.
        if vehicle.mode == "ALT_HOLD":
            # print "In Guided mode..."
            # print "Global Location (relative altitude): %s" % vehicle.location.global_relative_frame

            if (blueland == True):
                land()

            if (red1Good == True and red2Good == True) :
                yaw = bearing #* multYaw  # Set maximum yaw in degrees either side
                roll = offset * multRoll # Set maximum roll in degrees either side
                # Limit the range of roll possible
      
                pitch = 1500
                set_attitude (pitch, roll, yaw)
                if bearing >=10:
                    continue
                else:
                   vehicle.channels.overrides ={'1':1486,'2':1468,'3':1500}
                   time.sleep(0.1)
                   vehicle.flush()
                   #vehicle.channels.overrides = {'1':1500,'2':1500,'3':1500,'4':1500}
                   #time.sleep(0.1)
            else:
              vstate = "lost"
              
        else:
            # print "Exited GUIDED mode, setting tracking from following..."
            vstate = "tracking"
            
    return vstate
#------------------------------------------------------------------------------------
#----------------------------LOST Program--------------------------------------------
#-------------- FUNCTION DEFINITION TO FLY IN VEHICLE STATE LOST---------------------
def lost(vstate):
    print vstate
    print('LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
    #The vehicle process images and uses all data to fly in the lost state.
    # The vehicle rotates in one spot until a lock is established.
    # It sends attitude messages until manual control is resumed.
    maxPitch = -1
    multRoll= 0
    multYaw = 0
    yaw = roll = 0
    target = None # Initialise tuple returned from video stream
    found = False

    # Initialise the FPS counter.
    #fps = FPS().start()

    # Ready to land
    if vehicle.channels['5']>1700:
        land()

    while vstate =="lost":
        target = vs.readfollow()
        bearing = target[0]
        red1Good = target[1]
        offset = target[2]
        red2Good = target[3]
        circle = target[4]
        rlcontrol = target[5]
        moving_ratio = target[6]

        if vehicle.channels['5']>1700:
            land()
        
        # grab the frame from the threaded video stream and return left line offset
        # We do this to know if we have a 'lock' (goodTarget) as we come off of manual control.
        if rlcontrol == 1:
            pitch1=-1
            #roll1 = moving_ratio*0.3
            roll1 = 25
            print ('Lost mode check Following rlcontrol===> %d'%rlcontrol)
            set_attitude1(pitch1,roll1)
            #continue
        elif rlcontrol == 2:
            pitch1=-1
            #roll1= -moving_ratio*0.3
            roll1 = -35
            print ('Lost mode check Following rlcontrol===> %d'%rlcontrol)
            set_attitude1(pitch1,roll1)
            #continue
        elif rlcontrol == 3:
            pitch1=-1
            roll1=0
            print ('Lost mode check Following rlcontrol===> %d'%rlcontrol)
            set_attitude1(pitch1,roll1)
        else:
            print"No Line"

        if (red1Good ==True and red2Good ==True):
            print "Found"
            found = True
            
        else:
            target = vs.readlost()
            bearing = target[0]
            red1Good = target[1]
            roll = target[2]
            red2Good = target[3]
            circle = target[4]
            rlcontrol=target[5]
            moving_ratio=target[6]

                    
            if rlcontrol == 1:
                pitch1=-1
                #roll1= moving_ratio*0.3
                roll1 = 25
                print "Lost mode"
                print ('rlcontrol===> %d'%rlcontrol)
                set_attitude1(pitch1,roll1)
                if found!=True:
                   continue
            elif rlcontrol == 2:
                pitch1=-1
                #roll1= -moving_ratio*0.3
                roll1 = -35
                print "Lost mode"
                print ('rlcontrol===> %d'%rlcontrol)
                set_attitude1(pitch1,roll1)
                if found!=True:
                   continue
            elif rlcontrol == 3:
                pitch1=-1
                roll1=0
                print "Lost mode"
                print ('rlcontrol===> %d'%rlcontrol)
                set_attitude1(pitch1,roll1)
            else:
                print"No Line"
                 
            print 'Distance: %.3f'%(vehicle.rangefinder.distance)#sonar
            
        # Check if operator has transferred to autopilot using TX switch.
        if vehicle.mode == "ALT_HOLD":
            #print "In Guided mode, lost..."
                
            if found == True:
                #print "Found target - exiting lost state into following"
                vstate = "following"
            elif red1Good == True: # We have lock on upper roi.
                print "Lost -- upper lock"
            else: # We have nothing at all and need to rotate to look for something.
                # Set the attitude - note angles are in degrees
                print "Lost,STAY!!!"

            #vehicle.channels.overrides ={'1':1483,'2':1485,'3':1500}
            vehicle.channels.overrides ={'1':1492,'3':1500,'4':1450}
            time.sleep(0.2)
            vehicle.channels.overrides ={'1':1496,'3':1500,'4':1550}
            time.sleep(0.1)
            vehicle.flush()
            #vehicle.channels.overrides = {'1':1500,'2':1500,'3':1500,'4':1500}
            #time.sleep(0.1)
            
            print("MOVE-----Lost mode!!!")         
    return vstate
#-------------------------------------------------------------------------------------
#--- MAIN PROGRAM---#
#--------------------------SET UP CONNECTION TO VEHICLE----------------------------------
def main():
    vehicle = connect('/dev/serial0',baud=57600, wait_ready=True)
    time.sleep(0.5)
    print ('Basic pre-arm checks')
    # Parse the arguments
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(11,GPIO.OUT)
    pwm = GPIO.PWM(11,50)
    pwm.start(11.9)
    time.sleep(0.5)

    vehicle.mode = VehicleMode("STABILIZE")
    print("vehicle.mode:%s"%vehicle.mode)
    #vehicle.mode = VehicleMode("ALT_HOLD")
    vehicle.channels.overrides['3'] = 1100


    vehicle.armed = True
    time.sleep(0.5)
    print vehicle.armed


    while True:
        if vehicle.mode == VehicleMode("ALT_HOLD"):
            print("vehicle.mode:%s"%vehicle.mode)
            vehicle.mode = VehicleMode("ALT_HOLD")
            time.sleep(0.5)
            if vehicle.channels['5']>1700:
                land()
            break
        else:
            print('Waiting for program to engage')
            print(" Ch3: %s" % vehicle.channels['3'])
            print("vehicle.mode:%s"%vehicle.mode)
            vehicle.armed = True
            time.sleep(0.5)
            print vehicle.armed
            if vehicle.channels['5']>1700:
                land()


    vehicle.armed = True
    time.sleep(0.5)
    if vehicle.armed == True:
      print ('Arming motors')
      print vehicle.armed
    else:
      vehicle.armed = True
      time.sleep(1)
      print vehicle.armed
      print ('Arming motors')

    arm_and_takeoff(1.00)

    vstate = "tracking" # Set the vehicle state to tracking in the finite state machine.

    while True :

        if vstate == "tracking":
            if vehicle.channels['5']>1700:
                land()
            # Enter tracking state
            vstate = tracking(vstate)
            #print "Leaving tracking..."

        elif vstate == "following":
            if vehicle.channels['5']>1700:
                land()
            # Enter following state
            vstate = following(vstate)
            #print "Leaving following"
        else:
            if vehicle.channels['5']>1700:
                land()
            # Enter lost state
            vstate = lost(vstate)
            #print "Leaving lost"

    #Close vehicle object before exiting script
    print "Close vehicle object"
    vehicle.close()
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    main()