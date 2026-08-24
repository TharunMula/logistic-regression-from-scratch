from ucimlrepo import fetch_ucirepo
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# fetch dataset 
automobile = fetch_ucirepo(id=10)
print(automobile.data)
X = pd.DataFrame(automobile.data.features)
X = X.dropna(subset=['price'])

print("mean_price", X['price'].mean())
y=X['price']


features = [
    'engine-size',
    'horsepower',
    'highway-mpg'
]
X = X[features]
X['horsepower']= X['horsepower'].fillna(X['horsepower'].median())

# Create binary target
median_price = y.median()
y_logistic = (y >= median_price).astype(int)
print(X.isnull().sum())
print(y.isnull().sum())

#Spliting the train and test data now
x_train,x_test, y_train, y_test= train_test_split(
    X,y_logistic, test_size=0.2, random_state=42
)   

# Feature scaling
scale=StandardScaler()
scale.fit(x_train)
x_trained_scale= scale.transform(x_train)
x_tested_scale= scale.transform(x_test) 


w=np.zeros(x_train.shape[1])
b=0
m=len(x_trained_scale)
# sigmoid function:--
z=x_trained_scale @ w +b
def sigmoid(z):
    return  1/(1+np.exp(-z))
sigmoid_values = sigmoid(z)

#Logistic cost function
def compute_cost(x,y,w,b,m):
    cost=0
    z=x @ w +b
    predict=sigmoid(z)
    # cost function for logistic regression
    cost= -1/m * np.sum(y*np.log(predict) + (1-y)*np.log(1-predict))
    return cost
cost=compute_cost(x_trained_scale,y_train, w,b,m)
print(cost)


def compute_gradiant(x,y,w,b,m):
    dw=0
    db=0
    z=x @ w +b
    predict=sigmoid(z)
    error= predict-y
    cost=compute_cost(x,y, w,b,m)
    dw+=1/m*((x.T) @ error)
    db+=np.mean(error)
    return dw,db

dw,db=compute_gradiant(x_trained_scale,y_train,w,b,m) 
print(dw,db)


def gradiant_descent(x,y,w,b,m,l,ittr):
    cost_history=[]
    for i in range(ittr):
        dw,db= compute_gradiant(x,y,w,b,m)
        #updating the weight and bias
        w = w -(l * dw)
        b=b -(l* db)
        #calculating the cost after update
        cost=compute_cost(x,y,w,b,m)
        cost_history.append(cost)
    return w,b,cost_history
    
learning_rate=0.01
itteration=100
w,b,cost_history=gradiant_descent(x_trained_scale,y_train,w,b,m,learning_rate, itteration)
print("Training weight", w)
print("Training bias", b)
print("inital_cost", cost_history[0])
print("final_cost", cost_history[-1])


#probability calculation


probability= x_trained_scale @ w + b
probability= sigmoid(probability)

probability_train=(probability>0.5).astype(int)

print(probability_train[:10])

#test probability

z_test = x_tested_scale @ w + b

probability_test = sigmoid(z_test)

prediction_test = (probability_test >= 0.5).astype(int)

print("Probabilities:", probability_test[:10])
print("Predictions:", prediction_test[:10])



    
comparison = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Probability": probability_test,
    "Predicted": prediction_test
})

print(comparison.head(20))
# finding the test accuracy
accuracy = np.mean(
    prediction_test == y_test.to_numpy()
)

print("Test Accuracy:", accuracy)


#confusion matrix
actual = y_test.to_numpy()
predicted = prediction_test

TP = np.sum((actual == 1) & (predicted == 1))
TN = np.sum((actual == 0) & (predicted == 0))
FP = np.sum((actual == 0) & (predicted == 1))
FN = np.sum((actual == 1) & (predicted == 0))

print("True Positive:", TP)
print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)

#precision

precision = TP / (TP + FP)

print("Precision:", precision)

#recall

recall = TP / (TP + FN)

print("Recall:", recall)

#F1 score

f1 = 2 * (precision * recall) / (precision + recall)

print("F1 Score:", f1)


plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Logistic Regression Cost vs Iteration")
plt.show()






