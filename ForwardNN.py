import numpy as np
"2 layers softmax output, LeakyRelu in between"


def initialization(NeuronsNumber, InputNumber,OutputNumber,RandomScale):
    "Random Initialization"
    w01 = np.random.randn(NeuronsNumber, InputNumber) * RandomScale / np.sqrt(2.0/(InputNumber+NeuronsNumber))
    w12 = np.random.randn(OutputNumber, NeuronsNumber) * RandomScale / np.sqrt(NeuronsNumber)

    b1 = np.random.randn(NeuronsNumber, 1) * RandomScale / np.sqrt(2.0/(InputNumber+NeuronsNumber))
    b2 = np.random.randn(OutputNumber, 1) * RandomScale / np.sqrt(NeuronsNumber)
    return pack(w01,w12,b1,b2,NeuronsNumber, InputNumber,OutputNumber)

def training(NeuronsNumber, InputNumber,OutputNumber, Size ,X,Y,alpha,theta,momentum,momentumBeta,RMSprop,RMSpropBeta,RMSpropEPS):
    '''suppose 4 layers with equal number of neurons'''
    gradientcheck=False

    (w01, w12, b1, b2) = unpack(theta, NeuronsNumber, InputNumber, OutputNumber)

    "forward propagate"
    z1,a1,z2,a2=forwardpropagate(w01, w12, b1, b2, X)


    "Cost Function and Back Propagate"

    dz2=a2-Y
    db2=dz2
    db2 = np.sum(db2, axis=1,keepdims=True) / Size
    dw2=np.dot(dz2,np.transpose(a1))/Size

    da1=np.dot(np.transpose(w12),dz2)
    dz1=da1*activationfunction(z1,True)
    db1=dz1
    db1 = np.sum(db1, axis=1,keepdims=True) / Size
    dw1=np.dot(dz1,np.transpose(X))/Size


    "randomly triggerred grad check, not start from initial iteration"
    Check=np.random.randint(0,20)


    if gradientcheck and Check==0:

        "pack weights into theta"
        theta=pack(w01,w12,w23,w34,b1,b2,b3,b4,NeuronsNumber, InputNumber,OutputNumber)
        dtheta=pack(dw1,dw2,dw3,dw4,db1,db2,db3,db4,NeuronsNumber, InputNumber,OutputNumber)

        numerical_grad=np.zeros(theta.shape)

        for i in range(theta.shape[1]):
            addon=np.zeros(theta.shape)
            addon[0,i]+=0.00001
            upper=addon+theta
            addon[0,i] -= 0.00002
            lower= addon+theta
            (w01, w12, w23, w34, b1, b2, b3, b4)=unpack(upper,NeuronsNumber, InputNumber,OutputNumber)
            z1, a1, z2, a2, z3, a3, z4, a4 = forwardpropagate(w01, w12, w23, w34, b1, b2, b3, b4, X)
            lossupper = costfunction(a4, Y)
            (w01, w12, w23, w34, b1, b2, b3, b4) = unpack(lower, NeuronsNumber, InputNumber, OutputNumber)
            z1, a1, z2, a2, z3, a3, z4, a4 = forwardpropagate(w01, w12, w23, w34, b1, b2, b3, b4, X)
            losslower = costfunction(a4, Y)
            ratio=(lossupper-losslower)/(0.00002)
            numerical_grad[0,i]=ratio

        (test_w01, test_w12, test_w23, test_w34, test_b1, test_b2, test_b3, test_b4) = unpack(numerical_grad, NeuronsNumber, InputNumber, OutputNumber)

        print('Function Gradient', 'Numerical Gradient')
        for i in range(numerical_grad.shape[0]):
            print(numerical_grad[i,0], dtheta[i,0])
        diff = np.linalg.norm(numerical_grad - dtheta) / np.linalg.norm(numerical_grad + dtheta)
        print('Relative Difference: ')
        print(diff)
        assert(diff<0.0001)

    "update the weights"
    "add momentum methods and RMSprop---adam"
    dtheta=pack(dw1,dw2,db1,db2,NeuronsNumber, InputNumber,OutputNumber)

    momentum=momentumBeta*momentum+(1-momentumBeta)*dtheta
    RMSprop=RMSpropBeta* RMSprop+ (1-RMSpropBeta) *(np.power(dtheta,2))
    theta -= alpha * momentum/ np.sqrt(RMSprop+ RMSpropEPS)
    return theta,momentum,RMSprop


def pack(w01,w12,b1,b2,NeuronsNumber, InputNumber,OutputNumber):
    theta =w01.reshape(1,InputNumber*NeuronsNumber)
    theta =np.concatenate((theta,w12.reshape(1,OutputNumber*NeuronsNumber)),axis=1)

    theta =np.concatenate((theta,b1.transpose()),axis=1)
    theta =np.concatenate((theta, b2.transpose()),axis=1)
    return theta

def unpack(theta,NeuronsNumber, InputNumber,OutputNumber):
    number=0;
    temp=theta[0,number: number+InputNumber*NeuronsNumber]

    w01=np.reshape(temp,(NeuronsNumber,InputNumber))
    number +=InputNumber*NeuronsNumber


    w12 = np.reshape(theta[0,number:number+OutputNumber*NeuronsNumber],
                     (OutputNumber, NeuronsNumber))
    number+=OutputNumber*NeuronsNumber

    b1 = np.reshape(
        theta[0,number :
              number + NeuronsNumber * 1],
        (NeuronsNumber, 1))
    number += NeuronsNumber * 1
    b2 = np.reshape(
        theta[0,number :
              number + OutputNumber * 1],
        (OutputNumber, 1))
    return w01, w12, b1, b2

def forwardpropagate(w01,w12,b1,b2,X):
    z1 = np.dot(w01 , X) +b1
    a1 = activationfunction(z1)

    z2 = np.dot(w12 , a1) + b2
    a2 = softmax(z2)

    return z1,a1,z2,a2

def Lossfunction(output, expectedvalue):
    " The loss function of softmax output"
    Loss=np.copy(expectedvalue)
    Loss[expectedvalue==1]= -np.log(output[expectedvalue==1]+0.000000000000000000000000000001)
    Loss=np.sum(Loss,axis=0,keepdims=True)
    return Loss

def costfunction(output,expectedvalue):
    loss= Lossfunction(output,expectedvalue)
    loss=np.mean(loss,axis=1,keepdims=True)
    return loss

def activationfunction(input,gradient=False):
    if gradient == True:
        input[input < 0] = 0.01
        input[input >= 0] = 1
        return input
    input[input<0]=0.01*input[input<0]
    return input

def softmax(input):
    'prevent overflow by subtracting the maximum'
    input=input-np.max(input,axis=0,keepdims=True)
    input=np.exp(input)
    input=input/np.sum(input, axis=0,keepdims=True)
    return input



''' cost function works if value will not reach 1 or 0
expect=np.transpose(np.matrix([[1,1,0,0],[1,1,0,0]]))
value=np.transpose(np.matrix([[0.5,0.9,0.999999,0.0001],[0.6,0.9,0.999999,0.01]]))
print(expect.shape)
print(expect)
print(costfunction(value, expect,2))
'''

'''activationfunction works 
temp=np.ones((1,25))
print(activationfunction(temp))
'''

''' confirm the reshape function
a=np.matrix([1,2,3,4,5,6]).transpose()
print(a)
print(a.shape)
"devide into 2 rows"
a=a.reshape(2,3).transpose()
print(a)
print(a.shape)
'''






