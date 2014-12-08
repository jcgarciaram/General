
#readme file for CS8803 Final Project
## Summary
This readme document contains the following:
* An explanation of how to execute the program,
* A brief overview of the algorithm and justification for the selection, and
* A description of the source code and program functionality.

##Executing the Program

This program can use actual data that has already be collected in the `testing_video-centroid_data` file format. 

###Working with a List of Data


The first step to run the program is to process the collected data and generate the predicted data by running the module `PredictedDataFromKalmanFilter.py`. This module takes the file name as an argument in the following manner:
<pre><code>
python PredictedDataFromKalmanFilter.py FileName
</code></pre>

where FileName is the name of the file containing the observed data points. It then uses the Kalman filter algorithm to calculate the predicted data. Once this module has run, the file `PredictedDataFromKalmanFilter.py` will be created in the dir, and a `python matplot` will show a snap shot of the last observed data points and the 60 predicted data points. By default the plotting function is commented in the main function of `PredictedDataFromKalmanFilter.py`.

The final step and optional step to run the program is to compare the actual and predicted data by running the module `ComparingActualAndPredicted.py`. This module takes as an argument the actual data points. It then compares them against the points in the `Predicted_Centroid_Data.txt` and uses the data points to drive an animation to visualize the actual turtle(green circle) and predict turtle(red circle) sequentially. The L<sup>2</sup> error is printed in the console as the module is running. This module is run in the following manner:
<pre><code>
python ComparingActualAndPredicted.py FileName
</code></pre>

where FileName is a text file containing the actual data points.

##Algorithm Overview

The algorithm implemented in the program to estimate the next hexbug position is the Kalman filter. The Kalman filter was the first algorithm selected to be used in this project to estimate the hexbug position. The Kalman filter takes, as an input, the list of actual collected position data points and generates a prediction of what the next position data point will be after a period of time. The predictions are made in a manner where the error is statistically minimized. The results from the Kalman filter are satisfactory, and there was no need to implement another algorithm. An animation tool was added to the program to visualize how well the predicted data compares to the actual data. The animaion shows that the algorithm is operating with good accuracy and precision and is able to predict the hexbug's next position.

The Kalman filter algorithm was implemented in two phases, which are called update and predict. In the predict phase, a next position prediction is made using the current state estimation. In the update phase, the state is updated by incorporating the actual position in combination with the previous prediction. There is a four element state vector used in this implementation. The four elements represent the two locations (x and y) and two velocities (x and y velocities). The Kalman filter parameters and equations are implemented in the program as shown below. 

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

The Kalman filter is instrumental in predicting the next position after each actual measured position. The task is to predict the position once the actual measured positions are no longer available. To preform this task, the algorithm was extended with the use of a numeric physics model. The actual measured position data was analyized to determine the parameters needed to develop the model. The heart of the model is a ball-in-a-box with boundaries defined by the minium and maximum x and y dimensions. The typical ball-in-a-box model uses linear motion that reflects when there is a collision at the boundary. From observation, the motion of the hexbug is curvy in nature which can be predicted as radial motion; therefore, in order to have a better fit to the actual motion, the average radius of motion of the hexbug was calculated. The average radius was incorporated in the predicted simulated motion of the the hexbug. 


##Describe Source Code and Program Functionality

In the module `PredictedDataFromKalmanFilter.py`, the file given as an input is first read, and the positions of hexbug's centroid are stored in the `observed_x` and `observed_y` arrays. The module contains three functions called `kalmanXY()`, `kalman()`, and `predictKalmanXY()`. The `kalman()` function is where the Kalman filter algorithm is implemented. The `kalmanXY()` function works as a helper function by calling the `kalman()` function that setups the **F** (next state function) and **H** (measurement function) matrices. The **F** and **H** matrices have been defined specifically for the state vector used by the Kalman filter. The `predictKalmanXY()` function is where the list of actual position data points, `observed_x` and `observed_y`, are used by the Kalman filter algorithm to generate the predicted data points, `predicted_x_arr` and `predicted_y_arr`, and store them in the `Predicted_Centroid_Data.txt` file.

In order to predict the future data points the algorithm follows the following steps:
<ol>

<li>Calculate the average turning radius of all three consecutive points in the observed arrays. This is done by creating a circle with the three data points used as three points in the circle, and later calculating the radius of the circle.

<li>Once the average radius is calculated, calculate final turning radius of the observed data points using the last three data points captured. Using geometric functions, we then convert this circle into another circle with the avg radius that was calculated beforehand. However, the original direction of turning is kept by making the two circles be internally tangent.

<li>Because we now know the radius of the circle and the position of the last observed points in it, we can convert the points from cartesian coordinates to polar coordinates.

<li>A initial state vector with polar coordinates is then created with the polar points calculated: `x_polar`.

<li>The Kalman filter function is called using the polar state vector which then returns a prediction which follows the circle of average radius calculated. If the prediction falls outside the boundaries of the observable region, the direction of movement is then swapped so as to create the effect of the hexbug "bouncing" against the wall.

<li>The polar coordinates returned by the Kalman filter are later converted into cartesian coordinates and placed in the `predicted_x_arr` and `predicted_y_arr`.

<li>This process is repeated using the latest polar prediction as the current state vector.

<li> Once 60 points have been calculated, the predicted data points are written to the `Predicted_Centroid_Data.txt` file.
</ol>

After all of the predicted data points are generated, `python matplot` will display a snap shot comparing the actual and predicted data point.

In the module `ComparingActualAndPredicted.py`, the actual and predicted hexbug positions are displayed in an animation using the `python turtle` library. A text file comtaining the actual data points is input as an argument. The actual data pints and the predicted data points stored in `Predicted_Centroid_Data.txt` file are first read and stored as arrays. The hexbug's actual centroid data is stored as `x` and `y` arrays, and the predicted data points are stored as `x_predict` and `y_predict`. The module contains one function call `displayActualAndPredicted()`. This function is for the visualization of the actual and predicted position data points as an animation. The animation screen is setup. The turtles for the actual and predicted position visualizations are initialized. Then the actual and predicted data points are iterated and used to update the animation with the sequential position. During the iteration of the actual and predicted data points, the L<sup>2</sup> error is calculated and is printed in the console and updated as the animation is running.
