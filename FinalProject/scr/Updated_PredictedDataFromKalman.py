import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2


'''
The file with the actual hexbug centroid data points
is first read in and the hexbug's centroid data points
are stored as `observed_x` and `observed_y` arrays
'''



### Convert from cartesian to polar
def polar(x, y, deg=0):        # radian if deg=0; degree if deg=1
    from math import hypot, atan2, pi
    if deg:
        return hypot(x, y), 180.0 * atan2(y, x) / pi
    else:
        return hypot(x, y), atan2(y, x)

        
## Convert from polar to cartesian
def rect(r, w, deg=0):        # radian if deg=0; degree if deg=1
    from math import cos, sin, pi
    if deg:
        w = pi * w / 180.0
    return r * cos(w), r * sin(w)
    
"""This maps all angles to a domain of [-pi, pi]"""   
def angle_trunc(a):
    from math import pi
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi

f = open("All_Actual_Centroid_Data.txt", 'r')
observed_x = []
observed_y = []

cap = cv2.VideoCapture('/FinalProject/hexbug-training_video-transcoded.mp4')
    
video_width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

for line in csv.reader(f):
    #print line,'\t', line[0], '\t', line[1]
    line0_toString = str(line[0])
    line1_toString = str(line[1])
    x_str = line0_toString.strip('[')
    y_str = line1_toString.strip(']')
    observed_x.append(float(x_str))
    observed_y.append(video_height-float(y_str))
f.close()

#Find the Data Boundaries
observed_x_min = 5
observed_x_max = 0
observed_y_min = 0
observed_y_max = 0

supported_x_arr = []
supported_y_arr = []

predicted_x_arr = []
predicted_y_arr = []


print "OBSERVED:", observed_x_min,observed_x_max,observed_y_min,observed_y_max


'''
Adjusting the start variable will shift data set being used from the file to do the calculations.
So 60s at 24 fps is 1440 frames. 

The pastPoints variable is the number of points shown before the end of the of the 1440 positions. 
For instance with pastPoints set to 10 then points 1430 - 1440 of the actual data will be plotted
to show you what the actual data looked like before, the frames that need to be predicted.
'''
start = 10
end = 1440 + start
pastPoints = 10

def kalmanXY(x, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(4))):
    """
    Parameters:    
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise 
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H = np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.''')) 

def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty covariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)


    '''
    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x # measurement residual
    S = H * P * H.T + R  # residual covariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y          # Update state estimate
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P     #Updated estimate covariance

    # PREDICT x, P based on motion
    x = F*x + motion    #Predicted state estimate
    P = F*P*F.T + Q     #Predicted state covariance

    return x, P


def predictKalmanXY():
    from math import pi
    # initialize state
    x = np.matrix('0. 0. 0. 0.').T 
    P = np.matrix(np.eye(4))*1000 # initial uncertainty
    
  
    # plotting the actual observed data
    #plt.plot(observed_x, observed_y, 'go')
    
    # initialize for the position prediction with actual data available
    # initialize arrays for position prediction beyond actual data
    supportedPredictedResult = []
    futurePredictedResult = []
    
    # noise figure
    R = 0.01**2
    
    ctr = 0
    for meas in zip(observed_x, observed_y):
        # calling the Kalman filter algorithm on the actual
        # observed data to generate predictions.
        x, P = kalmanXY(x, P, meas, R)
        #print meas
        #print ((x[:2]).tolist()[0][0], (x[:2]).tolist()[1][0])
        
        supportedPredictedResult.append((x[:2]).tolist())
        ctr += 1
     
    
    supported_x, supported_y = zip(*supportedPredictedResult)
    plt.plot(supported_x, supported_y, 'b-')
    supported_x_arr = []
    supported_y_arr = []
    for i in range(len(supported_x)):
        supported_x_arr.append(supported_x[i][0])
        supported_y_arr.append(supported_y[i][0])
    
    x_p = x
    P_p = P
    '''
    TODO:
    Use the previous x and P for the predicting stage
    Fix the formatting of the previousPredictedResult 
    Get the plot to show all of the actual positions, the filter positions and the predicted future positions
    Make sure that the future positions begin to take place after the actual positions run out
    Make sure to add the boundary conditions to prevent the prediction to go out of bounds
    Clean up and refactor code
    '''
    '''
    # #Find the Data Boundaries
    x_min = min(x)/3
    x_max = max(x)/3
    y_min = min(y)/3
    y_max = max(y)/3
    distance_x = (x[2] - x[1])/3
    distance_y = (x[2] - x[1])/3
    for i in range(2000):
        #hexbug.hideturtle()
        #hexbug.penup()
        if ypos <= y_min:
            distance_y = (-1)*distance_y
        if xpos >= x_max:
            distance_x = (-1)*distance_x
        if ypos >= y_max:
            distance_y = (-1)*distance_y
        if xpos <= x_min:
            distance_x = (-1)*distance_x

        xpos = xpos + distance_x
        ypos = ypos + distance_y
    '''

    
    ##################################################
    #####Convert from cartesian to polar coordinates
    
    ##### Calculate center of circle of last turn
    last_three_x_coordinates = (observed_x[end], observed_x[end-2], observed_x[end-4])
    
    last_three_y_coordinates = (observed_y[end], observed_y[end-2], observed_y[end-4])
    
    A = np.array([last_three_x_coordinates[0], last_three_y_coordinates[0], 0.0])
    B = np.array([last_three_x_coordinates[1], last_three_y_coordinates[1], 0.0])
    C = np.array([last_three_x_coordinates[2], last_three_y_coordinates[2], 0.0])
    
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    
    s = (a + b + c) / 2
    
    Radius = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    
    P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3
    
    print "X:", last_three_x_coordinates
    print "Y:", last_three_y_coordinates
    print "P:", P
    print "Radius:", Radius
    
    ## Calculate coordinates of last three points in relation to center of circle
    A_turn = (A[0]-P[0],A[1]-P[1])
    B_turn = (B[0]-P[0],B[1]-P[1])
    C_turn = (C[0]-P[0],C[1]-P[1])
    
    # Finally, convert coordinates to polar
    A_polar = polar(A_turn[0],A_turn[1])
    B_polar = polar(B_turn[0], B_turn[1])
    C_polar = polar(C_turn[0], C_turn[1])
    
    print "A polar:", A_polar
    print "B_polar:", B_polar
    print "C_polar:", C_polar
    
    x_polar = np.matrix('0. 0. 0. 0.').T 
    x_polar[0] = A_polar[0]
    x_polar[1] = A_polar[1]
    x_polar[2] = A_polar[0]-B_polar[0]
    x_polar[3] = A_polar[1]-B_polar[1]
    ###############################
    #####END of polar conversion###
    
    
    xpos_previous = A[0] 
    ypos_previous = A[1]
    
    xpos_next = 0
    ypos_next = 0

    
    boundaryHit = False
    for i in range(60):
        if (i == 0):
            prediction_cartesian = (A[0], A[1])
            
            #Apply Kalman filter to predict based on Polar coordinates
            prediction_polar = (A_polar[0], A_polar[1]) 
            x_polar, P_p = kalmanXY(x_polar, P_p, prediction_polar, R)

            ## Convert to cartesian coordinates
            ## Calculate coordinates of x_cartesian in relation to origin (0,0)
            x_cartesian = rect(x_polar[0], x_polar[1])
            x_cartesian = (x_cartesian[0]+P[0],x_cartesian[1]+P[1])
          
            x_p[0], x_p[1] = x_cartesian[0], x_cartesian[1]
            
            
            xpos_next = (x_p[:2]).tolist()[0][0]
            ypos_next = (x_p[:2]).tolist()[1][0]
    
            distance_x = (xpos_next - xpos_previous)
            distance_y = (ypos_next - ypos_previous)
            
            print i, ": BEFORE: xpos_next, ypos_next, distance_X, distance_y", xpos_next, ypos_next, distance_x, distance_y
            
            ### Verify Whether Boundaries were hit
            if ypos_next <= observed_y_min:
                distance_y = (-1)*distance_y
                boundaryHit = True
                print i, "BOUNDARY 1"
            if xpos_next >= observed_x_max:
                distance_x = (-1)*distance_x
                boundaryHit = True
                print i, "BOUNDARY 2"
            if ypos_next >= observed_y_max:
                distance_y = (-1)*distance_y
                boundaryHit = True
                print i, "BOUNDARY 3"
            if xpos_next <= observed_x_min:
                distance_x = (-1)*distance_x
                boundaryHit = True
                print i, "BOUNDARY 4"
            
            
            xpos_next = xpos_previous + distance_x
            ypos_next = ypos_previous + distance_y
            print i, ": AFTER: xpos_next, ypos_next, distance_X, distance_y", xpos_next, ypos_next, distance_x, distance_y
         
            x_p[0] = xpos_next
            x_p[1] = ypos_next
            #x_p = np.matrix([[xpos_next, ypos_next, distance_x, distance_y]]).T 

            
            
        if (i >= 1):
            if boundaryHit == True:
                x_p, P_p = kalmanXY(x_p, P_p, prediction_cartesian, R)
            else:
                x_polar, P_p = kalmanXY(x_polar, P_p, prediction_polar, R)
                
                ## Account for differences in angle accross the pi line.
                if x_polar[1] > pi:
                    x_polar[1] = x_polar[1] - 2*pi
                if x_polar[1] < -pi:
                    x_polar[1] = x_polar[1] + 2*pi
                
                ## Convert new x_polar into cartesian coordinates
                x_cartesian = rect(x_polar[0], x_polar[1])
                x_cartesian = (x_cartesian[0]+P[0],x_cartesian[1]+P[1])

                x_p[0], x_p[1] = x_cartesian[0], x_cartesian[1]
                
                
            xpos_previous = prediction_cartesian[0]
            ypos_previous = prediction_cartesian[1]
             
            
            xpos_next = (x_p[:2]).tolist()[0][0]
            ypos_next = (x_p[:2]).tolist()[1][0]
    
            distance_x = (xpos_next - xpos_previous)
            distance_y = (ypos_next - ypos_previous)
            
            print i, ": BEFORE: xpos_next, ypos_next, distance_X, distance_y", xpos_next, ypos_next, distance_x, distance_y
            
            boundaryHit = False
            if ypos_next <= observed_y_min:
                distance_y = (-1)*distance_y
                print i, "BOUNDARY 1"
                boundaryHit = True
            if xpos_next >= observed_x_max:
                distance_x = (-1)*distance_x
                print i, "BOUNDARY 2"
                boundaryHit = True
            if ypos_next >= observed_y_max:
                distance_y = (-1)*distance_y
                print i, "BOUNDARY 3"
                boundaryHit = True
            if xpos_next <= observed_x_min:
                distance_x = (-1)*distance_x
                print i, "BOUNDARY 4"
                boundaryHit = True
            
            
            xpos_next = xpos_previous + distance_x
            ypos_next = ypos_previous + distance_y
            print i, ": AFTER: xpos_next, ypos_next, distance_X, distance_y", xpos_next, ypos_next, distance_x, distance_y
            
            x_p[0] = xpos_next
            x_p[1] = ypos_next
            #x_p = np.matrix([[xpos_next, ypos_next, distance_x, distance_y]]).T 
            
        prediction_cartesian = (xpos_next, ypos_next)
        
        ## Calculate coordinates of prediction in relation to center of circle previously calculated.
        ## and convert to polar    
        prediction_circle = (prediction_cartesian[0]-P[0],prediction_cartesian[1]-P[1])   
        prediction_polar = polar(prediction_circle[0],prediction_circle[1])
        
        print i, ": FINAL:", x_p[0], x_p[1], x_p[2], x_p[3]
        futurePredictedResult.append((x_p[:2]).tolist())
        
        
    predicted_x, predicted_y = zip(*futurePredictedResult)
    for i in range(len(predicted_x)):
        predicted_x_arr.append(predicted_x[i][0])
        predicted_y_arr.append(predicted_y[i][0])


def plotResults():
    # plotting the results
    plt.plot(observed_x[end + 2 - pastPoints:end + 60], observed_y[end + 2 - pastPoints:end + 60], 'g-o', label = 'Actual')
    plt.plot(observed_x[end+1], observed_y[end + 1], 'yo', label = 'ActualStart')
    plt.plot(supported_x_arr[end-96-pastPoints:], supported_y_arr[end-96-pastPoints:], 'b-o', label = 'Supported')
    plt.plot(predicted_x_arr, predicted_y_arr, 'r-o', label = 'Future')
    plt.plot(predicted_x_arr[0], predicted_y_arr[0], 'k-o', label = 'FutureStart')
    plt.legend( bbox_to_anchor=(0.5, 1.12), loc='upper center',  ncol = 5)
    plt.show()


def readInFile():
    global observed_x_min
    global observed_x_max
    global observed_y_min
    global observed_y_max
    
    '''
    The file with the actual hexbug centroid data points
    is first read in and the hexbug's centroid data points
    are stored as `observed_x` and `observed_y` arrays
    '''
    f = open("All_Actual_Centroid_Data.txt", 'r')
    
    for line in csv.reader(f):
        #print line,'\t', line[0], '\t', line[1]
        line0_toString = str(line[0])
        line1_toString = str(line[1])
        x_str = line0_toString.strip('[')
        y_str = line1_toString.strip(']')
        observed_x.append(float(x_str))
        observed_y.append(float(y_str))
    f.close()
    
    # #Find the Data Boundaries
    observed_x_min = min(observed_x)
    observed_x_max = max(observed_x)
    observed_y_min = min(observed_y)
    observed_y_max = max(observed_y)
    
    print "OBSERVED: ", observed_x_min, observed_x_max, observed_y_min, observed_y_max

def writeToFile():
    # The predicted data point generated by the Kalman filter
    # are written to a file
    fout = open('Predicted_Centroid_Data.txt','w')
    for i in range(len(predicted_x_arr)):
        fout.write(str(predicted_x_arr[i]) + ', ' + str(predicted_y_arr[i]) + '\n')
    fout.close()
    


readInFile()
print "OBSERVED: ", observed_x_min, observed_x_max, observed_y_min, observed_y_max
predictKalmanXY()  
plotResults() 




