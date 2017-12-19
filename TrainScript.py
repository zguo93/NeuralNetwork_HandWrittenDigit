import numpy as np
import matplotlib.pyplot as plt
from ForwardNN import initialization, pack,unpack,training, Lossfunction,costfunction,forwardpropagate



Size=60000
'''Check gradient checker works?'''


f=open('train-images-idx3-ubyte/train-images.idx3-ubyte','rb')
f.seek(16)
X=np.fromfile(f,dtype=np.uint8, count=28*28*Size)
X=X.reshape(Size,28*28).transpose()/100


"Demean the data"
"X=X-np.mean(X,axis=1,keepdims=True)"
f.close()




f=open('train-images-idx3-ubyte/train-labels.idx1-ubyte','rb')
f.seek(8)
array=np.fromfile(f,dtype=np.uint8, count=Size)
Y=np.zeros((Size,10))
"print(array)"
for i in range(0,Size):
    num=array[i]
    Y[i][num]=1
"print(Y)"
Y=Y.transpose()
f.close()


Test_size=10000
'''The test images and labels'''
f=open('train-images-idx3-ubyte/t10k-images.idx3-ubyte','rb')
f.seek(16)
test_x=np.fromfile(f,dtype=np.uint8, count=28*28*Test_size)
test_x=test_x.reshape(Test_size,28*28).transpose()/100

"Demean the data"
"test_x=test_x-np.mean(X,axis=1,keepdims=True)"
f.close()

f=open('train-images-idx3-ubyte/t10k-labels.idx1-ubyte','rb')
f.seek(8)
array=np.fromfile(f,dtype=np.uint8, count=Test_size)
test_y=np.zeros((Test_size,10))
"print(array)"
for i in range(0,Test_size):
    num=array[i]
    test_y[i][num]=1
"print(Y)"
test_y=test_y.transpose()
f.close()



'Start to run training and save the error '
'hyperparameter'
neuronNumber=1500
OutputNumber=10
InputNumber=28*28
stepsize=64
alpha=0.005
momentumBeta=0.9
RMSpropBeta=0.999
RMSpropEPS=0.000000001


'basic parameters'
theta=initialization(neuronNumber,28*28,10,1)
momentum=np.zeros(theta.shape)
RMSprop=np.zeros(theta.shape)


'test parameters'
test_error=[]
train_error=[]
test_accuracy=0
list_test_accuracy=[]
list_train_accuracy=[]
best_accuracy=0
best_train_accuracy=0


i=0

for iteration in range(0,100000,1):
    input=np.array(X[:,i:i+stepsize])
    output=np.array(Y[:,i:i+stepsize])
    i+=stepsize
    if i>Size:
        i=0
    "print(i)"
    "print(output)"

    "Monitor weights change"
    previous_theta=theta


    theta,momentum,RMSprop=training(neuronNumber,InputNumber,OutputNumber,stepsize,input,output,alpha/np.sqrt(iteration+1),theta,momentum,momentumBeta,RMSprop,RMSpropBeta,RMSpropEPS)


    if iteration %100 ==0:
        print('The iteration '+str(iteration))
        w01, w12, b1, b2 = unpack(theta, neuronNumber, 28 * 28, 10)
        z1, a1, z2, a2 = forwardpropagate(w01, w12, b1, b2, X)
        train_cost = costfunction(a2, Y)
        train_accuracy = np.equal(np.argmax(a2, 0), np.argmax(Y, 0))
        train_accuracy = np.sum(train_accuracy) / len(train_accuracy)
        'store the accuracy into list'
        print(train_accuracy)
        list_train_accuracy=np.append(list_train_accuracy,train_accuracy)
        if train_accuracy>best_train_accuracy:
            best_train_accuracy=train_accuracy

        z1, a1, z2, a2 = forwardpropagate(w01, w12, b1, b2, test_x)
        error = costfunction(a2, test_y)
        'the test accuracy'
        test_accuracy = np.equal(np.argmax(a2, 0), np.argmax(test_y, 0))
        test_accuracy = np.sum(test_accuracy) / len(test_accuracy)
        print(test_accuracy)
        'store the accuracy into list'
        list_test_accuracy = np.append(list_test_accuracy, test_accuracy)
        if test_accuracy> best_accuracy:
            best_accuracy=test_accuracy


    displayInfo=False
    if displayInfo:
        print("The weigth change :")
        print(np.linalg.norm(previous_theta-theta))
        w01, w12, b1, b2=unpack(theta,neuronNumber,28*28,10)



        "Calculate the loss function and accuracy"
        'the test error'
        z1, a1, z2, a2=forwardpropagate(w01,w12,b1,b2,test_x[:,0:2000])
        error=costfunction(a2,test_y[:,0:2000])
        'the test accuracy'

        test_accuracy=np.equal(np.argmax(a2,0),np.argmax(test_y[:,0:2000],0))
        test_accuracy=np.sum(test_accuracy)/2000
        'the training error'
        z1, a1, z2, a2 = forwardpropagate(w01, w12, b1, b2, X[:,0:2000])
        train_cost = costfunction(a2, Y[:,0:2000])

        train_accuracy = np.equal(np.argmax(a2, 0), np.argmax(Y[:,0:2000], 0))
        train_accuracy = np.sum(train_accuracy) / len(train_accuracy)

        print("The result :")
        print(test_accuracy)
        print(train_accuracy)
        print(error)

        print()
        train_error=np.append(train_error,error)
        test_error=np.append(test_error,error)

print (test_error)

'Plot the graph'

plt.figure(1)
plt.plot(list_train_accuracy)
plt.title('The training set accuracy ')
plt.show()

plt.figure(2)
plt.plot(list_test_accuracy)
plt.title('The test set accuracy ')
plt.show()


"Save the data of trained network"
s='2layers'+str(neuronNumber)
np.save(s,theta)

"The result of neural network"
w01, w12, b1, b2=unpack(theta,neuronNumber,28*28,10)
"the training error"
z1, a1,z2,a2 = forwardpropagate(w01,w12, b1,b2, X)

accuracy=np.equal(np.argmax(a2,0),np.argmax(Y,0))
accuracy=np.sum(accuracy)/Size
print('The training accuracy :')
print(accuracy)

"the test error"

z1, a1,z2,a2 = forwardpropagate(w01,w12, b1,b2, test_x)

accuracy=np.equal(np.argmax(a2,0),np.argmax(test_y,0))
accuracy=np.sum(accuracy)/len(accuracy)
print('The test accuracy :')
print(accuracy)

print('The best train accuracy:')
print(best_train_accuracy)

print('The best test accuracy:')
print(best_accuracy)