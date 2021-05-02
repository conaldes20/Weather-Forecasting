import numpy as np
import csv
import math
#import predict as util
from datetime import datetime

class DeepNNetwork:
    def __init__(self):

        # load the dataset from the CSV file
        reader = csv.reader(open("C:/Users/CONALDES/Documents/weather_features.csv", "r"), delimiter=",")
        x = list(reader)
        
        print("OOJAOBAPOWER_SinglePoint_Interannual")
        print("                              ")
        print("### Input Data ###")
        print("==================")
        for row in x:
            print(', '.join(row))

        #print("np.array(x[2:]): " + str(np.array(x[2:])))
        self.features = np.array(x[2:]).astype("float")

        # temperature/relative humidity and heat index are splitted. 1 is appended at each temperature/relative humidity pair for the bias
        self.data_x = np.concatenate((self.features[:,:2], np.ones((self.features.shape[0],1))), axis=1)
        self.data_y = self.features[:,2:]

        print("                              ")
        print("### Weather Attributes (data_x with biases) and Heat Index (data_y) ###")
        print("=======================================================================")
        print("data_x: " + str(self.data_x))
        print("data_y: " + str(self.data_y))        
        print("                              ")
        
        celcius = self.data_x[:,0]
        print("celcius: " + str(celcius))
        mean_celcius = np.mean(celcius)
        std_celcius = np.std(celcius)
        for row in range(len(self.data_x)):
            temp = (self.data_x[row, 0] - mean_celcius)/std_celcius
            self.data_x[row, 0] = temp

        rh2m = self.data_x[:,1]
        mean_rh2m = np.mean(rh2m)
        std_rh2m = np.std(rh2m)
        for row in range(len(self.data_x)):
            temp = (self.data_x[row,1] - mean_rh2m)/std_rh2m
            self.data_x[row, 1] = temp

        self.min_htindex = np.min(self.data_y)
        self.max_htindex = np.max(self.data_y)
        for row in range(len(self.data_x)):
            temp = (self.data_y[row] - self.min_htindex)/(self.max_htindex - self.min_htindex)
            self.data_y[row] = temp
        
        # dataset metadata for the prediction part of the network are mean_celcius, std_celcius, mean_rh2m, std_rh2m, self.min_htindex, self.max_htindex
        #self.predict = util.Predict(mean_celcius, std_celcius, mean_rh2m, std_rh2m, self.min_htindex, self.max_htindex)
        self.predict = self.saveDataSetMetadata(mean_celcius, std_celcius, mean_rh2m, std_rh2m, self.min_htindex, self.max_htindex)

        # we set a threshold at 80% of the data
        self.m = float(self.features.shape[0])
        self.m_train_set = int(self.m * 0.8)
        
        print("### Traning Set (80%) and Testing Set (20%) ###")
        print("===============================================")
        print("m_train_set: " + str(roundup(self.m_train_set,0)))
        print("m_test_set: " + str(roundup((self.m - self.m_train_set),0)))                   
        print("                              ")

        # we split the train and test set using the threshold
        self.x, self.x_test = self.data_x[:self.m_train_set,:], self.data_x[self.m_train_set:,:]
        self.y, self.y_test = self.data_y[:self.m_train_set,:], self.data_y[self.m_train_set:,:]

        print("### Normalized Traning Data (x with biases) ###")
        print("===============================================")
        print("x: " + str(self.x))
        print("y: " + str(self.y))                   
        print("                              ")
        print("### Normalized Testing Data (x_test with biases) ###")
        print("====================================================")
        print("x_test: " + str(self.x_test))
        print("y_test: " + str(self.y_test))                       
        print("                              ")

        # we init the network parameters
        self.z2, self.a2, self.z3, self.a3, self.z4, self.a4 = (None,) * 6
        self.delta2, self.delta3, self.delta4 = (None,) * 3
        self.djdw1, self.djdw2, self.djdw3 = (None,) * 3
        self.gradient, self.numericalGradient, self.chkedgradt = (None,) * 3
        self.Lambda = 0.1   # For regularization
        self.learning_rate = 0.1

        #parameters
        inputSize = 2
        hidden1Size = 3
        hidden2Size = 2
        outputSize = 1        

        # init weights
        np.random.seed(0)
        WT1 = np.random.rand(inputSize, hidden1Size)    # (2x3) weight matrix from input to hidden layer
        WT2 = np.random.rand(hidden1Size, hidden2Size)  # (3x2) weight matrix from hidden to output layer
        WT3 = np.random.rand(hidden2Size, outputSize)   # (2x1) weight matrix from hidden to output layer

        ww1 = np.append(WT1, [[0.1, 0.1, 0.1]], axis=0)
        ww2 = np.append(WT2, [[0.1, 0.1]], axis=0)
        ww3 = np.append(WT3, [[0.1]], axis=0)
        
        # convert weights to matrix
        self.w1 = np.matrix(ww1)
        self.w2 = np.matrix(ww2)
        self.w3 = np.matrix(ww3)
        
        print("### Weights Generated (with biases) ###")
        print("=======================================")
        print("w1: " + str(self.w1))
        print("w2: " + str(self.w2))
        print("w3: " + str(self.w3))
        print("                              ")
        
    def forward(self):

        # first layer
        self.z2 = np.dot(self.x, self.w1)        
        self.a2 = np.tanh(self.z2)

        # we add the 1 unit (bias) at the output of the first layer
        ba2 = np.ones((self.x.shape[0], 1))
        self.a2 = np.concatenate((self.a2, ba2), axis=1)

        # second layer
        self.z3 = np.dot(self.a2, self.w2)
        self.a3 = np.tanh(self.z3)

        # we add the 1 unit (bias) at the output of the second layer
        ba3 = np.ones((self.a3.shape[0], 1))
        self.a3 = np.concatenate((self.a3, ba3), axis=1)

        # output layer, prediction of our network
        self.z4 = np.dot(self.a3, self.w3)
        self.a4 = np.tanh(self.z4)
        #print("self.a4: " + str(self.a4))
        
    # back propagation with regularisation
    def backward(self):
        
        # gradient of the cost function with regards to W3
        self.delta4 = np.multiply(-(self.y - self.a4), tanh_prime(self.z4))
        self.djdw3 = ((self.a3.T * self.delta4) / self.m_train_set) + self.Lambda * self.w3/self.m_train_set
        
        # gradient of the cost function with regards to W2
        self.delta3 = np.multiply(self.delta4 * self.w3.T, tanh_prime(np.concatenate((self.z3, np.ones((self.z3.shape[0], 1))), axis=1)))
	#np.delete(self.delta3, 2, axis=1) removes the bias term from the backpropagation
        self.djdw2 = ((self.a2.T * np.delete(self.delta3, 2, axis=1)) / self.m_train_set) + self.Lambda * self.w2/self.m_train_set
        
        # gradient of the cost function with regards to W1
	#np.delete(self.delta3, 2, axis=1) removes the bias term from the backpropagation
        self.delta2 = np.multiply(np.delete(self.delta3, 2, axis=1) * self.w2.T, tanh_prime(np.concatenate((self.z2, np.ones((self.z2.shape[0], 1))), axis=1)))
        #np.delete(self.delta2, 3, axis=1) removes the bias term from the backpropagation
        self.djdw1 = ((self.x.T * np.delete(self.delta2, 3, axis=1)) / self.m_train_set) + self.Lambda * self.w1/self.m_train_set
        #print("self.a3.T, self.delta4, (self.a3.T * self.delta4) : " + str(self.a3.T.shape) + ", " +  str(self.a3.T.shape) + ", " +  str((self.a3.T * self.delta4).shape))
        #print("self.delta4, self.w3.T, (self.delta4 * self.w3.T) : " + str(self.delta4.shape) + ", " +  str(self.w3.T.shape) + ", " + ", " +  str((self.delta4 * self.w3.T).shape))
        #print("w2: " + str(self.w2))

        
    def update_gradient(self):
        # division by self.m_train_set taken care of in back propagation
        # Gradient descent learning rule. Weight (w) rescaled by a factor (1 - (self.Lambda * self.learning_rate)/self.m_train_set) called weight decay.         
        
        self.w1 += - self.learning_rate * (self.djdw1 + self.Lambda * self.w1/self.m_train_set)
        self.w2 += - self.learning_rate * (self.djdw2 + self.Lambda * self.w2/self.m_train_set)
        self.w3 += - self.learning_rate * (self.djdw3 + self.Lambda * self.w3/self.m_train_set)

    def cost_function(self):
        # quadratic cost function plus sum of the squares of all the weights in the network.
        return (0.5 * sum(np.square((self.y - self.a4))))/self.m_train_set + ((0.5*self.Lambda) / self.m_train_set) * (
            np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)) + np.sum(np.square(self.w3)))

    def set_weights(self, weights):
        self.w1 = np.reshape(weights[0:9], (3, 3))
        self.w2 = np.reshape(weights[9:17], (4, 2))
        self.w3 = np.reshape(weights[17:20], (3, 1))

    def compute_gradients(self):
        # reval is a real value of complex number
        self.gradient = np.concatenate((self.djdw1.ravel(), self.djdw2.ravel(), self.djdw3.ravel()), axis=1).T

    def compute_numerical_gradients(self):
        weights = np.concatenate((self.w1.ravel(), self.w2.ravel(), self.w3.ravel()), axis=1).T

        self.numericalGradient = np.zeros(weights.shape)
        perturbation = np.zeros(weights.shape)
        
        e = 1e-4
        #print("Weights size: " + str(len(perturbation)))
        #print("================")
        #print("weights: " + str(weights))
        for p in range(len(perturbation)):
            # Set perturbation vector
            perturbation[p,0] = e	#All elements of perturbation take 0.0001 one after the other in the iteration cycle
            #print("perturbation: " + str(p) + ", " + str(perturbation))            
            self.set_weights(weights + perturbation)
            #print("weights + perturbation: " + str(p) + ", " + str(weights + perturbation))
            self.forward()
            loss2 = self.cost_function()

            self.set_weights(weights - perturbation)
            #print("weights - perturbation: " + str(p) + ", " + str(weights - perturbation))
            self.forward()
            loss1 = self.cost_function()
            
            self.numericalGradient[p,0] = (loss2 - loss1) / (2 * e)   #[[0.07324063]]
            #print("weight disturbed " + str(p + 1) + ": (index " + str(p) + ") ")
            #print("weights:     weights + perturbation:   weights - perturbation:   numericalGradient:")
            #print(str(roundup(weights[p,0],8)) + "       " + str(roundup((weights + perturbation)[p,0],8)) + "                  " + str(
            #    roundup((weights + perturbation)[p,0],8)) + "            " + str(roundup(self.numericalGradient[p,0],8)))

            perturbation = np.zeros(weights.shape)            

        self.set_weights(weights)   # Reset weights to normal, i.e. without perturbation        

    def check_gradients(self):
        self.compute_gradients()
        self.compute_numerical_gradients()
        #print("                             ")
        #print("Vector of elements of gradients computed during backpropagation (V1) and ")
        #print("Vector of elements of numerical gradients (V2) compared: ")
        #print("       V1,                      V2")
        #for p in range(len(self.gradient)):
        #    print(str(self.gradient[p,0]) + ",      " + str(self.numericalGradient[p,0]))
        
        self.chkedgradt = np.linalg.norm(self.gradient - self.numericalGradient) / np.linalg.norm(
            self.gradient + self.numericalGradient)
        
        #print("                             ")
        print("Gradient checked: " + str(self.chkedgradt))

    #def predict(self, X):
    #    self.x = X
    #    self.forward()
    #    return self.a4

    def saveDataSetMetadata(self, mean_celcius, std_celcius, mean_rh2m, std_rh2m, min_heatIndex, max_heatIndex):
        self.mean_celcius = mean_celcius
        self.std_celcius = std_celcius
        self.mean_rh2m = mean_rh2m
        self.std_rh2m = std_rh2m
        self.min_heatIndex = min_heatIndex
        self.max_heatIndex = max_heatIndex

    def input(self, celcius, rh2m):
        celcius = (celcius - self.mean_celcius) / self.std_celcius
        rh2m = (rh2m - self.mean_rh2m) / self.std_rh2m
        return np.matrix([[
            celcius, rh2m
        ]])

    def output(self, heatIndex):
        return heatIndex * (self.max_heatIndex - self.min_heatIndex) + self.min_heatIndex

    def r2(self):
        y_mean = np.mean(self.y)
        ss_res = np.sum(np.square(self.y - self.a4))
        ss_tot = np.sum(np.square(self.y - y_mean))
        return 1 - (ss_res / ss_tot)

    def summary(self, step):
        print("Loss %f" % (self.cost_function()))
        print("RMSE: " + str(np.sqrt(np.mean(np.square(self.a4 - self.y)))))
        print("MAE: " + str(np.sum(np.absolute(self.a4 - self.y)) / self.m_train_set))
        print("R2: " + str(self.r2()))

    def predict_heatIndex(self, celcius, rh2m):
        #self.x = np.concatenate((self.predict.input(celcius, rh2m), np.ones((1, 1))), axis=1)
        self.x = np.concatenate((self.input(celcius, rh2m), np.ones((1, 1))), axis=1)
        nn.forward()
        #print("Predicted Heat Index (Celcius): " + str(roundup(self.predict.output(self.a4[0,0]), 2)))
        #print("Predicted Heat Index (Celcius): " + str(roundup(self.output(self.a4[0,0]), 2)))
        print("Predicted Heat Index (Celcius): " + str(self.output(self.a4[0,0])))

def roundup(a, digits=0):
    #n = 10**-digits
    #return round(math.ceil(a / n) * n, digits)
    return round(a, digits)

def tanh_prime(x):
    return 1.0 - np.square(np.tanh(x))

nn = DeepNNetwork()

# current date and time
now = datetime.now()
starting_time = now.strftime("%H:%M:%S")
timestamp1 = datetime.timestamp(now)

done = False

nb_it = 100000
for step in range(nb_it):
    nn.forward()
    nn.backward()
    nn.update_gradient()

    nn.check_gradients()
    if nn.chkedgradt < 5.25e-07:
        done = True
        now = datetime.now()
        stopping_time = now.strftime("%H:%M:%S")
        print("                              ")
        print("Back Propagation computes correctly the gradients:")
        print("==================================================")
        print("If gradient checked is in the order of 10e-07, it is a great approximation")
        print("Iteration: %d, " % (step + 1))
        print("Starting Time = ", starting_time)
        print("Stopping Time = ", stopping_time)
        timestamp2 = datetime.timestamp(now)
        print("Time Elapsed = " + str(roundup((timestamp2 - timestamp1), 2)) + "secs")
        print("                             ")
        break

if not done:    
    now = datetime.now()
    stopping_time = now.strftime("%H:%M:%S")
    print("                              ")
    print("Iteration: %d, " % (step + 1))
    print("Starting Time = ", starting_time)
    print("Stopping Time = ", stopping_time)
    timestamp2 = datetime.timestamp(now)
    print("Time Elapsed = " + str(roundup((timestamp2 - timestamp1), 2)) + "secs")
    print("                             ")

print("### Weights After Training (with biases) ###")
print("============================================")
print("w1: " + str(nn.w1))
print("w2: " + str(nn.w2))
print("w3: " + str(nn.w3))
print("                              ")

print("### Computed Output (y), Network Generated Output (a4) and Difference ###")
print("=========================================================================")
print("   y        (a4)      (y - a4)")
print("==============================")
for i in range(nn.m_train_set):
    nn.y[i,0] = (nn.max_htindex - nn.min_htindex)*nn.y[i,0] + nn.min_htindex
    nn.a4[i,0] = (nn.max_htindex - nn.min_htindex)*nn.a4[i,0] + nn.min_htindex
for i in range(nn.m_train_set):    
    #print("  " + str(roundup(nn.y[i,0], 2)) + "     " + str(roundup(nn.a4[i,0],2)) + "       " + str(roundup((nn.y[i,0]-nn.a4[i,0]),2)))
    print("  " + str(nn.y[i,0]) + "     " + str(nn.a4[i,0]) + "       " + str((nn.y[i,0]-nn.a4[i,0]))) 
print("                              ")

nn.x = nn.x_test
nn.y = nn.y_test
nn.forward()

print("### Testing summary ###")
print("=======================")
nn.summary(nb_it)
print("                              ")
print("### Predict ###")
print("===============")

celcius = 0.00
rh2m = 0.00

while True:
    while True:
        try:
            celcius = float(input("Enter Temperature (Celcius) or 0 to quit: "))
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
    if celcius == 0.00:
        break    
     
    while True:
        try:
            rh2m = float(input("Enter Relative Humidity (%) or 0 to quit: "))
            break
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
    if rh2m == 0.00:
         break
    
    print("                              ")
    entry_list = [celcius, rh2m]
    print("entry_list: " + str(entry_list))
    nn.predict_heatIndex(celcius, rh2m)	
    print("                              ")
