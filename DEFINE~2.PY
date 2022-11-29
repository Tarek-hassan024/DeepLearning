__author__ = 'WEI'
import tensorflow as tf
import math
import numpy as np
pi=math.pi


def shrink_soft_threshold(r,rvar,theta,K):

    theta=theta[0]
    lam=theta*tf.sqrt(rvar)
    lam = tf.maximum(tf.abs(lam), 0)
    ampl=tf.maximum(tf.abs(r)-lam,0)
    xhat=r*tf.cast(ampl/tf.abs(r),tf.complex128)
    dxdr=tf.reduce_mean(tf.cast((tf.abs(r)-lam)>0,tf.complex128)*tf.cast((1-lam/(2*tf.abs(r))),tf.complex128),0)
    dxdr1=tf.reduce_mean(tf.cast((tf.abs(r)-lam)>0,tf.complex128)*tf.cast((0.5*tf.cast(lam,tf.complex128)*tf.sqrt(r)*tf.pow(tf.conj(r),-3/2)),tf.complex128),0)
    return (xhat,dxdr,dxdr1)



def shrink_gmest(corr, rvar, theta, K):

    th=tf.constant(-40., dtype=tf.float64)

    P = tf.cast(tf.exp(theta[0]) / tf.reduce_sum(tf.exp(theta[0])), tf.complex128)
    mu = tf.cast(theta[1], tf.complex128)
    var = tf.cast(tf.exp(tf.maximum(th, theta[2])), tf.complex128)
    U = 0
    V = 0
    dU = 0
    dV = 0
    dU1= 0
    dV1 =0
    rvar=tf.cast(rvar,tf.complex128)
    for k in range(K):
        with tf.name_scope('tilde_mu_k_x'):
            tilde_mu_k_x = (mu[k] * rvar + corr * var[k]) / (rvar + var[k])
        with tf.name_scope('CNpdf_k'):
            CNpdf_k = tf.cast(tf.exp(tf.maximum(th, tf.real(
                -tf.conj(corr - mu[k]) * (corr - mu[k]) / 2 / (rvar + var[k])))) / 2 / np.pi / tf.real(var[k] + rvar),
                              tf.complex128)
        with tf.name_scope('u_k'):
            u_k = tilde_mu_k_x * P[k] * CNpdf_k
        with tf.name_scope('du_k'):
            du_k = var[k] / (rvar + var[k]) * P[k] * CNpdf_k + u_k * (-tf.conj(corr - mu[k]) / 2 / (rvar + var[k]))
        with tf.name_scope('du_k1'):
            du_k1=u_k * (-(corr - mu[k]) / 2 / (rvar + var[k]))
        with tf.name_scope('v_k'):
            v_k = P[k] * CNpdf_k
        with tf.name_scope('dv_k'):
            dv_k = v_k * (-tf.conj(corr - mu[k]) / 2 / (rvar + var[k]))
        with tf.name_scope('dv_k1'):
            dv_k1 = v_k * (-(corr - mu[k]) / 2 / (rvar + var[k]))
        U = U + u_k
        dU = dU + du_k
        dU1 = dU1 +du_k1
        V = V + v_k
        dV = dV + dv_k
        dV1 = dV1 +dv_k1

    xhat = U / V
    dxdr = (dU * V - U * dV) / (V * tf.conj(V))
    dxdr1= (dU1*V-U*dV1)/(V*tf.conj(V))
    dxdr = tf.reduce_mean(dxdr, 0)
    dxdr1= tf.reduce_mean(dxdr1,0)


    return (xhat, dxdr, dxdr1)


def get_shrinkage_function(name, K):

    gm_theta = np.array([np.ones(K) / K, np.zeros(K), -np.ones(K)/1000000000]).astype(np.float64)

    try:
        return {
            'soft':(shrink_soft_threshold,(1,1)),
            'gm':(shrink_gmest,gm_theta),
        }[name]
    except KeyError as ke:
        raise ValueError('unrecognized shrink function %s' % name)
        sys.exit(1)
