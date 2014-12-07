import numpy as np
import cv2



x = []
y = []

def collectData():
    '''
    This function reads in the video file using OpenCV then using the 
    OpenCV toolbox processes the video file to get the xy data points of 
    hexbug's centroid. The centroid data point are stored in a text file. 
    '''
    
    # Create file handle for the video file
    cap = cv2.VideoCapture('hexbug-training_video-transcoded.mp4')
    
    print "Video Properties:"
    print "\t Width: ",cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    print "\t Height: ",cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    print "\t FourCC: ",cap.get(cv2.cv.CV_CAP_PROP_FOURCC)
    print "\t Framerate: ",cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print "\t Number of Frames: ",cap.get(7)
    
    # created file handle for data text file
    fout = open('All_Actual_Centroid_Data.txt','w')
    
    # define range of color in HSV
    hmin = (238-30)/2
    hmax = (238+30)/2
    smin = 255/2
    smax = 255
    vmin = int(0.20 * 255)
    vmax = int(1.0 * 255)
    lower = np.array([hmin,smin,vmin])
    upper = np.array([hmax,smax,vmax])
    
    
    ctr = 0
    offset = 10
    numFramesCaptured = 50
    Start = True
    while Start:
        ctr += 1
        if ctr % 100 == 0:
            print ctr
        if ctr == (numFramesCaptured+offset):
            Start = False
            
        
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, lower, upper)
        #cv2.imshow('mask',mask)
        
        # Applies a fixed-level threshold to each array element
        _,thresh = cv2.threshold(mask,127,255,0)
        
        # Finds contours in a binary image
        contours,_ = cv2.findContours(thresh, 1, 2)
        
        if contours:
            cnt = contours[0]
            # Calculates all of the moments up to the third order of a polygon 
            # or rasterized shape
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = float(M['m10']/M['m00'])
                cy = float(M['m01']/M['m00'])
                fout.write(str(cx) + ', ' + str(cy) + '\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    fout.close()
    cap.release()
    cv2.destroyAllWindows()
    
# Calling the collectData() function  
collectData()    
  
    
