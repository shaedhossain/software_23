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

print("Hello, Git! Updated version.")
print("Hello, Git! Updated version.")
print("Hello, Git! Updated version.")
print("Hello, Git! Updated version.")
print("new branch new check")
