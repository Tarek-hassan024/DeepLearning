__author__ = 'WEI'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!

import define_networks
from scipy.io import loadmat




if __name__ == '__main__':


    N = 256
    M = 128
    type = 'ULA'  #'UPA'
    shrink = 'soft'
    MN = str(N)+str(M)
    SNRrange = '10to20dB' # '0to10dB'
    vSNRrange = '10dB'
    T = 8  # the number of the layers
    D = loadmat('CSmatrix'+MN+'.mat')
    A = D['A']


    save_network = 'Trained_LAMP_for_SV_'+ type +'_'+ MN + SNRrange +'_'+ vSNRrange + '.npz'
    tc = 'SV_'+type+'_training_channel.mat'
    tm = 'SV_'+type+'_training_'+str(M)+'_measurements_'+SNRrange+'.mat'

    vc = 'SV_'+type+'_validation_channel.mat'
    vm = 'SV_'+type+'_validation_'+str(M)+'_measurements_'+vSNRrange+'.mat'


    layers, y_ = define_networks.build_LAMP(A, T=T, shrink=shrink, untied=True, K=1)
    training_stages, x_ = define_networks.setup_training(A, layers=layers, trinit=1e-3, refinements=(.5, .1, .01))
    sess = define_networks.do_training(x_=x_, y_=y_, training_stages=training_stages, savefile=save_network, tc=tc, tm=tm, vc=vc, vm=vm, maxit=1000000)


