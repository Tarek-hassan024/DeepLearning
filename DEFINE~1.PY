__author__ = 'WEI'

import define_shrinkage
import tensorflow as tf
import numpy as np
import numpy.linalg as la
import sys
from scipy.io import loadmat



def build_LAMP(A,T,shrink,untied,K):
    eta,theta_init = define_shrinkage.get_shrinkage_function(shrink,K=K)
    layer=[]
    var_all=[]
    A_=tf.constant(A, name='A',dtype=tf.float64)
    M,N=A.shape
    B=A.T/(1.01*la.norm(A,2)**2)
    B_=tf.Variable(B,dtype=tf.float64,name='B_0')
    var_all.append(B_)

    y_ = tf.placeholder(tf.complex128, (M, None))

    yreal_=tf.real(y_)
    yimag_=tf.imag(y_)
    #The first layer: v=y
    Byreal_=tf.matmul(B_,yreal_)
    Byimag_=tf.matmul(B_,yimag_)
    By_=tf.complex(Byreal_,Byimag_)
    theta_=tf.Variable(theta_init,dtype=tf.float64,name='theta_0')
    var_all.append(theta_)

    OneOverM = tf.constant(float(1)/M,dtype=tf.float64)
    NOverM = tf.constant(float(N)/M,dtype=tf.complex128)
    rvar_= OneOverM*tf.reduce_sum(tf.square(tf.abs(y_)),0)

    xhat_,dxdr_,dxdr1_= eta(By_,rvar_,theta_,K=K)

    layer.append(('LAMP-{0} linear T=1'.format(shrink),By_,(B_,),tuple(var_all),(0,)))
    layer.append(('LAMP-{0} non-linear T=1'.format(shrink),xhat_,(theta_,),tuple(var_all),(1,)))

    v_=y_
    for t in range(1,T):

        b_=NOverM*dxdr_
        c_=NOverM*dxdr1_
        Axreal_=tf.matmul(A_,tf.real(xhat_))
        Aximag_=tf.matmul(A_,tf.imag(xhat_))
        Ax_=tf.complex(Axreal_,Aximag_)
        v_=y_-Ax_+b_*v_+c_*tf.conj(v_)
        temp=tf.abs(v_)
        rvar_=OneOverM*tf.reduce_sum(temp*temp,0)
        theta_=tf.Variable(theta_init,dtype=tf.float64,name='theta_'+str(t))
        var_all.append(theta_)

        if untied:
            B_=tf.Variable(B,dtype=tf.float64,name='B_'+str(t))
            Bvreal_=tf.matmul(B_,tf.real(v_))
            Bvimag_=tf.matmul(B_,tf.imag(v_))
            Bv_=tf.complex(Bvreal_,Bvimag_)
            rhat_=xhat_+Bv_
            var_all.append(B_)
            layer.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,),tuple(var_all),(0,) ) )
        else:
            Bvreal_=tf.matmul(B_,tf.real(v_))
            Bvimag_=tf.matmul(B_,tf.imag(v_))
            Bv_=tf.complex(Bvreal_,Bvimag_)
            rhat_=xhat_+Bv_

        xhat_,dxdr_,dxdr1_=eta(rhat_,rvar_,theta_,K=K)
        layer.append(('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,),tuple(var_all),(1,)))
    return layer,y_
def save_trainable_vars(sess,filename,**kwargs):

    save={}
    for v in tf.trainable_variables():
        save[str(v.name)]=sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

def load_trainable_vars(sess,filename):
    other={}
    try:
        tv=dict([(str(v.name),v)for v in tf.trainable_variables()])
        for k,d in np.load(filename,allow_pickle=True).items():
            if k in tv:
                print('restore '+ k)
                sess.run(tf.assign(tv[k],d))
                #print(sess.run(tv[k]))
            else:
                other[k]=d
                #print('error!')
    except IOError:
        pass
    return other



def setup_training(A,layers,trinit=1e-3,refinements=(.5,.1,.01),final_refine=None):

    training_stages=[]
    M,N=A.shape
    x_ = tf.placeholder(tf.complex128, (N, None))
    for name,xhat_,var_list,var_all,flag in layers:

        #loss_=tf.nn.l2_loss(tf.abs(xhat_-prob.x_))
        #nmse_=tf.nn.l2_loss(tf.abs(xhat_-prob.x_))/tf.nn.l2_loss(tf.abs(prob.x_))
        #loss_=tf.reduce_mean(tf.reduce_sum(tf.abs(xhat_-prob.x_),axis=0)
        nmse_=tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(xhat_ - x_)), axis=0)) / tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(x_)), axis=0))
        loss_=nmse_  
        print(var_list)
        if var_list is not None:
            if flag==(0,):
                train_=tf.train.AdamOptimizer(trinit).minimize(loss_,var_list=var_list)
                training_stages.append((name,xhat_,loss_,nmse_,train_,var_list,var_all,flag))
            else:
                train_=tf.train.AdamOptimizer(trinit).minimize(loss_,var_list=var_list)
                training_stages.append((name,xhat_,loss_,nmse_,train_,var_list,var_all,flag))
        index=0
        for fm in refinements:
            train2_=tf.train.AdamOptimizer(fm*trinit).minimize(loss_,var_list=var_all)
            training_stages.append((name+' trainrate='+str(index),xhat_,loss_,nmse_,train2_,(),var_all,flag))
            index=index+1

    return training_stages,x_

def assign_trainable_vars(sess,var_list,var_list_old):
    for i in range(len(var_list)):

        temp=sess.run(var_list_old[i])
        print(temp)
        sess.run(tf.assign(var_list[i],temp))

def do_training(x_,y_,training_stages,savefile,tc,tm,vc,vm,iv1=10,maxit=1000000,better_wait=5000):

    Dtx=loadmat(tc)
    xt=Dtx['x']
    Dty=loadmat(tm)
    yt=Dty['y']

    trainingsize=np.size(xt,axis=1)

    Dvx=loadmat(vc)
    xv=Dvx['x']
    Dvy=loadmat(vm)
    yv=Dvy['y']


    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    state=load_trainable_vars(sess,savefile)

    done=state.get('done',[])
    log=str(state.get('log',[]))
    layernmse=state.get('layernmse',[])

    var_list_old0=()
    var_list_old1=()
    nmse_dB=None
    
    for name,xhat_,loss_,nmse_,train_,var_list,var_all,flag in training_stages:
        if name in done:
            if name=='LAMP-gm linear T=5':
                var_list_old0=var_list
            if name=='LAMP-gm non-linear T=5':
                var_list_old1=var_list
            print('Already did  '+ name + ' skipping.')
            continue
        if len(var_list):
            print(name+' '+ 'extending '+ ','.join([v.name for v in var_list]))
            if flag==(0,): #if linear operation
                if nmse_dB is not None:
                    #nmse_dB=round(nmse_dB,6)
                    layernmse=np.append(layernmse,nmse_dB)
                    print(layernmse)
                if len(var_list_old0):
                #Initialize the training variable to the value of that in previous layer
                    assign_trainable_vars(sess,var_list,var_list_old0)
                    print(var_list_old0)
                var_list_old0=var_list
            else :
                
                if len(var_list_old1):
                    assign_trainable_vars(sess,var_list,var_list_old1)
                    print(var_list_old1)
                var_list_old1=var_list
        else:
            print(name+' '+'fine tuning all ' + ','.join([v.name for v in var_all]))
        nmse_history=[]
        for i in range(maxit+1):
            if i%iv1==0:
                nmse=sess.run(nmse_,feed_dict={x_:xv,y_:yv}) #validation results
                nmse=round(nmse,5)
                if np.isnan(nmse):
                    raise RuntimeError('nmse is Nan')
                nmse_history=np.append(nmse_history,nmse)
                nmse_dB=10*np.log10(nmse)
                nmsebest_dB=10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(iv1*100)==0:
                    print('')
                    age_of_best=len(nmse_history)-nmse_history.argmin()-1
                    if age_of_best*iv1>better_wait:
                        break
            rand_index=np.random.choice(trainingsize,size=128)
            x=xt[...,rand_index]
            y=yt[...,rand_index]
            sess.run(train_,feed_dict={x_:x,y_:y})

        done=np.append(done,name)
        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done']=done
        state['log']=log
        state['layernmse']=layernmse

        save_trainable_vars(sess,savefile,**state)
        # for t in [1,2,3,4,5,6,7,8,9,10,11,12]:
        #     if name=='LAMP-gm non-linear T='+str(t)+' trainrate='+str(2):
        #         save_trainable_vars(sess, savefile+'T='+str(t)+'.npz', **state)

    layernmse=np.append(layernmse,nmse_dB)
    print(layernmse)
    state['layernmse']=layernmse
    save_trainable_vars(sess,savefile,**state)
    return sess

