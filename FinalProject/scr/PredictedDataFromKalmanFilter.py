import numpy as np
import matplotlib.pyplot as plt
import csv

'''
The file with the actual hexbug centroid data points
is first read in and the hexbug's centroid data points
are stored as `observed_x` and `observed_y` arrays
'''
f = open("Actual_Centroid_Data.txt", 'r')
observed_x = []
observed_y = []
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
    # initialize state
    x = np.matrix('0. 0. 0. 0.').T 
    P = np.matrix(np.eye(4))*1000 # initial uncertainty
    
  
    # plotting the actual observed data
    plt.plot(observed_x, observed_y, 'go')
    
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

    xpos_previous = 0
    ypos_previous = 0
    
    xpos_next = 0
    ypos_next = 0
 
    for i in range(60):
        if (i == 0):
            prediction = (supported_x_arr[len(supported_x_arr)-1], supported_y_arr[len(supported_y_arr)-1])
            xpos_previous = prediction[0]
            ypos_previous = prediction[1]
            x_p, P_p = kalmanXY(x_p, P_p, prediction, R)
            xpos = (x_p[:2]).tolist()[0][0]
            ypos = (x_p[:2]).tolist()[1][0]
            
        if (i >= 1):
            xpos_previous = prediction[0]
            ypos_previous = prediction[1]
             
            x_p, P_p = kalmanXY(x_p, P_p, prediction, R)
            
            xpos_next = (x_p[:2]).tolist()[0][0]
            ypos_next = (x_p[:2]).tolist()[1][0]
    
            distance_x = (xpos_next - xpos_previous)
            distance_y = (ypos_next - ypos_previous)
            
            if ypos <= observed_y_min:
                distance_y = (-1)*distance_y
            if xpos >= observed_x_max:
                distance_x = (-1)*distance_x
            if ypos >= observed_y_max:
                distance_y = (-1)*distance_y
            if xpos <= observed_x_min:
                distance_x = (-1)*distance_x
    
            xpos = xpos + distance_x
            ypos = ypos + distance_y
    
            
        prediction = (xpos, ypos)
  
        
        futurePredictedResult.append((x_p[:2]).tolist())
    predicted_x, predicted_y = zip(*futurePredictedResult)

      
    # The predicted data point generated by the Kalman filter
    # are written to a file
#     fout = open('Predicted_Centroid_Data.txt','w')
#     for i in range(len(kalman_x)):
#         fout.write(str(kalman_x[i][0]) + ', ' + str(kalman_y[i][0]) + '\n')
#     fout.close()
    
    plt.plot(predicted_x, predicted_y, 'r-')
    plt.show()


predictKalmanXY()





