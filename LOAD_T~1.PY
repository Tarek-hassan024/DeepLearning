from numpy import load
import numpy as np
import scipy.io as sio

K = 4
T = 8
type = 'UPA'  # 'UPA'
network = 'GM_LAMP'  # LAMP
channel='SV' # DeepMIMO
term='_K=' + str(K) #''

MN = '256128'
SNRrange = '0to10dB'  # '10to20dB'
vSNRrange= '10dB'

trainedfilename = 'Trained_'+network+'_for_'+channel+'_' + type + '_' + MN + SNRrange + '_' + vSNRrange +term+ '.npz'
print(trainedfilename)
data = load(trainedfilename)
savefilename = 'Trained_'+network+'_for_'+channel+'_' + type +'_'+MN+SNRrange+'.mat'

item='B_0:0'
B_0=data[item]
item='B_1:0'
B_1=data[item]
item='B_2:0'
B_2=data[item]
item='B_3:0'
B_3=data[item]
item='B_4:0'
B_4=data[item]
item='B_5:0'
B_5=data[item]
item='B_6:0'
B_6=data[item]
item='B_7:0'
B_7=data[item]

item='theta_0:0'
theta0=data[item]
item='theta_1:0'
theta1=data[item]
item='theta_2:0'
theta2=data[item]
item='theta_3:0'
theta3=data[item]
item='theta_4:0'
theta4=data[item]
item='theta_5:0'
theta5=data[item]
item='theta_6:0'
theta6=data[item]
item='theta_7:0'
theta7=data[item]


# print(theta0,theta1,theta2,theta3,theta4,theta5,theta6,theta7)

sio.savemat(savefilename, {'B_0': B_0,'B_1': B_1,'B_2': B_2,'B_3': B_3,'B_4': B_4,'B_5':B_5,'B_6': B_6,'B_7':B_7,'theta0':theta0,'theta1':theta1,'theta2':theta2,'theta3':theta3,'theta4':theta4,'theta5':theta5,'theta6':theta6,'theta7':theta7})


print('save to '+savefilename)
