
#readme file for CS8803 Final Project
## Summary
This readme document contains the following:
* An explanation of how to execute the program
* A brief overview of the algorithm and justification for the selection
* A describe source code and program functionality

##Executing the Program

This program can extract actual data from a video file or can use actual data that has already be collected in the `testing_video-centroid_data` file format.
### Working with Video File
The first step is to collect centroid data from the video file by running the module `ActualDataFromVideo.py`. Make sure that the video is in the same directory (now to be referred to as " dir ") and that the `cap = cv2.VideoCapture("path\videofile")` call contains the right path for the video file. It is also important, to set the `numFramesCaptured` variable to capture the desired number of frames. The `Actual_Centroid_Data.txt` file will be created in the dir after the module has run. 

The next step is to process the collected data and generate the predicted data by running the module `PredictedDataFromKalmanFilter.py`. This module takes `Actual_Centroid_Data.txt` as an input then uses the Kalman filter algorithm to calculate the predicted data. Once this module has run the `Predicted_Centroid_Data.txt` file will be created in the dir and a `python matplot` will show a snap shot comparing the actual and predicted data point.

The final step is to compare the actual and predicted data by running the module `ComparingActualAndPredicted.py`. This module takes both `Actual_Centroid_Data.txt` and `Predicted_Centroid_Data.txt` as inputs and uses the data points to drive an animation to visualize the actual turtle(green circle) and predict turtle(red circle) sequentially. The L<sup>2</sup> error is printed in the console as the module is running.  

###Working with a List of Data
When the actual centroid data which has already been collect as a list of data points the first step described in the **Working with Video File** section can be skipped. 

The next step is to process the collected data and generate the predicted data by running the module `PredictedDataFromKalmanFilter.py`. This module takes `Actual_Centroid_Data.txt` as an input then uses the Kalman filter algorithm to calculate the predicted data. Once this module has run the `Predicted_Centroid_Data.txt` file will be created in the dir and a `python matplot` will show a snap shot comparing the actual and predicted data point.

The final step is to compare the actual and predicted data by running the module `ComparingActualAndPredicted.py`. This module takes both `Actual_Centroid_Data.txt` and `Predicted_Centroid_Data.txt` as inputs and uses the data points to drive an animation to visualize the actual turtle(green circle) and predict turtle(red circle) sequentially. The L<sup>2</sup> error is printed in the console as the module is running. 

##Algorithm Overview

The algorithm implemented in the program to estimate the next hexbug position is the Kalman filter. The Kalman filter was the first algorithm used in this project to estimate the hexbug position. The Kalman filter takes as an input the list of actual collected position data points and generates a prediction of what the next position data point will after a period of time. The prediction are made in a manner where the error is statistically minimized. The results from the Kalman filter are very satisfactory and there was no need to implement another algorithm. An animation tool was added to the program to visualize how well the predicted data compares to the actual data. The animaion shows that the algorithm is operating with good accuracy and precisious and able to predict the hexbug's next position. The calculations show that the L<sup>2</sup> error after comparing approximately 60 frames of actual and predicted data was 27.

The Kalman filter algorithm was implemented as two phase that are called update and predict that operate in a cycle. In the predict phase, a next position prediction is made using the current state estimation. In the update phase, the state is updated by incorporating the actual position in combination with the previous prediction. There is a four element state vector used in this implementation. The four elements represent the two locations(x and y) and two velocities (x and y velocities). The Kalman filter parameters and equations are implemented in the program as show below. 

~~~python
    '''
    Parameters:
    x: initial state
    P: initial uncertainty conariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x
    
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
~~~



##Describe Source Code and Program Functionality

In the module `ActualDataFromVideo.py` there is a function called `collectData()`. This function reads in the video file using OpenCV then using the OpenCV toolbox processes the video file to get the x and y position of the hexbug's centroid. The centroid positions are stored in a text file call `Actual_Centroid_Data.txt` as a list of data points. 

In the module `PredictedDataFromKalmanFilter.py` the `Actual_Centroid_Data.txt` file is first read in and the positions of hexbug's centroid are stored in the `observed_x` and `observed_y` arrays. The module contains three functions called `kalmanXY()`, `kalman()`, and `predictKalmanXY()`. The `kalman()` function is where the Kalman filter algorithm is implemented. The `kalmanXY()` function works as a helper function by calling the `kalman()` function that setups the **F** (next state function) and **H** (measurement function) matrices. The **F** and **H** matrices have been defined specifically for the state vector used by the Kalman filter. The `predictKalmanXY()` function is where the list of actual position data points (`observed_x` and `observed_y`) are used by the Kalman filter algorithm to generate the predicted data points (`kalman_x` and `kalman_y`) and store them in the `Predicted_Centroid_Data.txt` file. After all of the predicted data points are generated `python matplot` will show a snap shot comparing the actual and predicted data point.

In the module `ComparingActualAndPredicted.py` the actual and predicted hexbug positions are displayed in an animation using the `python turtle` library. The `Actual_Centroid_Data.txt` and `Predicted_Centroid_Data.txt` files are first read in and stored as arrays. The hexbug's actual centroid data is stored as `x` and `y` arrays and the predicted data points are stored as `x_predict` and `y_predict`. The module contains one function call `displayActualAndPredicted()`. This function is for the visualization of the actual and predicted position data points as an animation. The animation screen is setup. The turtles for the actual and predicted position visualization are initialized. Then the actual and predicted data points are iterated through and used to update the animation with the sequential position. During the iteration of the actual and predicted data points the L<sup>2</sup> error is calculated and is printed in the console and updated as the animation is running. 









