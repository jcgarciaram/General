import csv
from cmath import sqrt

'''
The file with the actual and predicted hexbug centroid data
points are first read in and the actual hexbug's centroid data
points are stored as `x` and `y` arrays, and the predicted data
points are stored as `x_predicted` and `y_predicted` arrays.
'''

f = open("Actual_Centroid_Data.txt", 'r')
x = []
y = []
for line in csv.reader(f):
    #print line,'\t', line[0], '\t', line[1]
    line0_toString = str(line[0])
    line1_toString = str(line[1])
    x_str = line0_toString.strip('[')
    y_str = line1_toString.strip(']')
    x.append(float(x_str))
    y.append(float(y_str))
f.close()


f = open("Predicted_Centroid_Data.txt", 'r')
x_predict = []
y_predict = []
for line in csv.reader(f):
    #print line,'\t', line[0], '\t', line[1]
    line0_toString = str(line[0])
    line1_toString = str(line[1])
    x_str = line0_toString.strip('[')
    y_str = line1_toString.strip(']')
    x_predict.append(float(x_str))
    y_predict.append(float(y_str))
f.close()
    

def displayActualAndPredicted():
    #For Visualization
    import turtle
    det = 0

    # Animation Screen Setup
    window = turtle.Screen()
    window.bgcolor('white')
    
    # Initializing the turtles for the actual and 
    # predicted position visualization
    hexbug_predicted = turtle.Turtle()
    hexbug_predicted.shape('circle')
    hexbug_predicted.color('red')
    hexbug_predicted.resizemode('user')
    hexbug_predicted.shapesize(0.1, 0.1, 0.1)
    hexbug_predicted.goto(x[0]/3, y[0]/3)

  
    hexbug_actual = turtle.Turtle()
    hexbug_actual.shape('circle')
    hexbug_actual.color('green')
    hexbug_actual.resizemode('user')
    hexbug_actual.shapesize(0.1, 0.1, 0.1)
    hexbug_actual.goto(x[0]/3, y[0]/3)

    # Iterating through actual and predicted data and updating 
    # the animation with the sequential position
    for i in range(len(x)):

        hexbug_actual.goto(x[i]/3, y[i]/3)
        hexbug_actual.showturtle()
        hexbug_actual.pendown()
  
        hexbug_predicted.goto(x_predict[i]/3, y_predict[i]/3)
        hexbug_predicted.showturtle()
        hexbug_predicted.pendown()
        
        #L^2 Error Calculation
        det = det + (((x_predict[i]/3) - (x[i]/3))**2 + ((y_predict[i]/3) - (y[i]/3))**2)
        L_2 = sqrt(det)
        print L_2
        
   
displayActualAndPredicted() 