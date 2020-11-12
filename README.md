We use Pixhawk as our flight controller for our quadrotor, and Raspberry Pi to process our code. 
Our goal is to allow the quadrotor to takeoff automatically and follow the black line track. 
The quadrotor when land automatically once it detect the blue area at the end of the track. 

1. PiVideoStream.py
   Divide the image we read from the pi camera into two parts: top and down, as our ROI (Region of Interest). 
   Then, turn the RGB image to HSV image and set "Black" as our target color to recognize. 
   Therefore, we can find the largest black area in each ROI. 
   After that, we take its center of mass for our coordinate and calculate the angle between the two. So, the quadrotor knows where its heading.

2. quadrotor.py
   The quadrotor will first raise its throttle level to takeoff. Once it reaches a specific height it will stay and start to detect the track. 
   The quadrotor will land when it detect the blue landing area.
