import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


x_train= np.loadtxt('files/train_X.csv', delimiter=',').T
y_train= np.loadtxt('files/train_label.csv',delimiter=',').T

x_test= np.loadtxt('files/test_X.csv', delimiter=',').T
y_test= np.loadtxt('files/test_label.csv',delimiter=',').T

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

def tanh(x):
  return np.tanh(x)

def relu(x):
  return np.maximum(x,0)

def softmax(x):
  expp= np.exp(x)
  return expp/np.sum(expp,axis=0)

def derivative_tanh(x):
  return (1-np.power(x,2))

def derivative_relu(x):
  return np.array(x>0, dtype=np.float32)

def initialize_para(n_x, n_h,n_y):
  w1=np.random.randn(n_h,n_x)*0.01
  b1=np.zeros((n_h,1))

  w2=np.random.randn(n_y,n_h)*0.01
  b2=np.zeros((n_y,1))

  parameters= {'w1':w1,
               'b1':b1,
               'w2':w2,
               'b2':b2}
  return parameters

def forward(x, parameters):
  w1=parameters['w1']
  b1=parameters['b1']
  w2=parameters['w2']
  b2=parameters['b2']

  z1=np.dot(w1,x)+b1

  a1=tanh(z1)

  z2=np.dot(w2,a1)+b2

  a2=softmax(z2)

  cache={'z1':z1,
         'a1':a1,
         'z2':z2,
         'a2':a2}
  return  cache

def cost(a2,y):
  m=y.shape[1]
  cost= -(1/m)*np.sum(np.sum(y*np.log(a2),1))
  return cost


print("Hello, Git! Updated version.")
print("Hello, Git! Updated version.")
print("Hello, Git! Updated version.")
print("Hello, Git! Updated version.")
print("again into new_b branch.")

def backpro( x,y , parameters, forward):
  w1= parameters['w1']
  b1= parameters['b1']
  w2= parameters['w2']
  b2= parameters['b2']

  a1=forward['a1']
  a2=forward['a2']

  m= x.shape[1]
  
  dz2=(a2-y)
  dw2=(1/m)*np.dot(dz2,a1.T)
  db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)

  dz1=(1/m)*np.dot(w2.T,dz2)*derivative_tanh(a1)
  dw1=(1/m)*np.dot(dz1,x.T)
  db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)

  grad= {
      'dw1':dw1,
      'db1':db1,
      'dw2':dw2,
      'db2':db2
  }
  return grad


def update_para(parameters, grad, learning_rate):
  w1=parameters['w1']
  b1=parameters['b1']
  w2=parameters['w2']
  b2=parameters['b2']

  dw1=grad['dw1']
  db1=grad['db1']
  dw2=grad['dw2']
  db2=grad['db2']

  w1=w1-learning_rate*dw1
  b1=b1-learning_rate*db1
  w2=w2-learning_rate*dw2
  b2=b2-learning_rate*db2

  parameters= {'w1':w1,
               'b1':b1,
               'w2':w2,
               'b2':b2}
  return parameters
