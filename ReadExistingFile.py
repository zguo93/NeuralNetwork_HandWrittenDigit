import numpy as np
import matplotlib.pyplot as plt
from ForwardNN import initialization, pack,unpack,training, Lossfunction,costfunction,forwardpropagate

print('Retrain the neural network')

Size=60000


f=open('train-images-idx3-ubyte/train-images.idx3-ubyte','rb')
f.seek(16)
X=np.fromfile(f,dtype=np.uint8, count=28*28*Size)
X=X.reshape(Size,28*28).transpose()


"Demean the data"
X=X-np.mean(np.mean(X,axis=1),axis=0)
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


Test_size=6000
'''The test images and labels'''
f=open('train-images-idx3-ubyte/t10k-images.idx3-ubyte','rb')
f.seek(16)
test_x=np.fromfile(f,dtype=np.uint8, count=28*28*Test_size)
test_x=test_x.reshape(Test_size,28*28).transpose()

"Demean the data"
test_x=test_x-np.mean(np.mean(X,axis=1),axis=0)
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


theta=np.load('2layers1500.npy')

'Start to run training and save the error '
cost=[]
neuronNumber=300
stepsize=100
for i in range(0,60000,stepsize):
    input=np.array(X[:,i:i+stepsize])
    output=np.array(Y[:,i:i+stepsize])
    "print(i)"
    "print(output)"

    "Monitor weights change"
    previous_theta=theta


    theta=training(neuronNumber,28*28,10,stepsize,input,output,0.0001/(np.sqrt(i+1)),theta)

    print("The weigth change :")
    print(np.linalg.norm(previous_theta-theta))

    w01, w12, b1, b2=unpack(theta,neuronNumber,28*28,10)
    z1, a1, z2, a2=forwardpropagate(w01,w12,b1,b2,test_x[:,0:6000])
    error=costfunction(a2,test_y[:,0:6000])
    temp=np.argmax(a2,0)
    accuracy=np.equal(np.argmax(a2,0),np.argmax(test_y[:,0:6000],0))
    accuracy=np.sum(accuracy)/6000
    print("The result :")
    print(accuracy)
    print(error)
    print()
    cost=np.append(cost,error)
print (cost)

s='2layers'+str(neuronNumber)
np.save(s,theta)