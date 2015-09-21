#!/usr/bin/env python
# encoding: utf-8
r"""
Riemann solvers for the Euler equations

This module contains Riemann solvers for the Euler equations which have the
form (in 1d):

.. math:: 
    q_t + f(q)_x = 0
  
where 

.. math:: 
    q(x,t) = \left [ \begin{array}{c} \rho \\ \rho u \\ E \end{array} \right ],
  
the flux function is 

.. math:: 
    f(q) = \left [ \begin{array}{c} \rho u \\ \rho u^2 + p \\ u(E+p) \end{array}\right ].

and :math:`\rho` is the density, :math:`u` the velocity, :math:`E` is the 
energy and :math:`p` is the pressure.

Unless otherwise noted, the ideal gas equation of state is used:

.. math::
    E = (\gamma - 1) \left (E - \frac{1}{2}\rho u^2 \right)

:Authors:
    Kyle T. Mandli (2009-06-26): Initial version
    Kyle T. Mandli (2011-03-28): Interleaved version
"""
# ============================================================================
#      Copyright (C) 2009 Kyle T. Mandli <mandli@amath.washington.edu>
#
#  Distributed under the terms of the Berkeley Software Distribution (BSD) 
#  license
#                     http://www.opensource.org/licenses/
# ============================================================================

import numpy as np
import numba
from math import sqrt
import time


num_eqn = 3
use_numba = True

def euler_roe_1D(q_l, q_r, aux_l, aux_r, gamma1, efix):
    r"""
    Roe Euler solver in 1d

    *aug_global* should contain -
     - *gamma* - (float) Ratio of the heat capacities
     - *gamma1* - (float) :math:`1 - \gamma`
     - *efix* - (bool) Whether to use an entropy fix or not

    See :ref:`pyclaw_rp` for more details.

    :Version: 1.0 (2009-6-26)
    """

    # Problem dimensions
    num_rp = q_l.shape[1]
    num_waves = 3

    # Calculate Roe averages
    u, a, enthalpy = roe_averages(q_l, q_r, gamma1)[0:3]
    # Find eigenvector coefficients
    a1, a2, a3 = compute_eigenvector_coeff(q_l, q_r, a, u, enthalpy, gamma1)

    # Compute the waves
    wave, s = compute_waves(a, a1, a2, a3, enthalpy, u, num_waves, num_rp)

    # Entropy fix
    if efix:
        raise NotImplementedError("Entropy fix has not been implemented!")
    else:
        amdq, apdq = godunov_update(num_rp, num_eqn, num_waves, wave, s)

    return wave, s, amdq, apdq

@numba.jit(nopython=True, cache=True)
def godunov_update_compiled(num_rp, num_eqn, num_waves, wave, s):
    # Godunov update
    amdq = np.zeros((num_eqn, num_rp))
    apdq = np.zeros((num_eqn, num_rp))
    s_index = np.zeros((2,num_rp))
    for m in xrange(num_eqn):
        for mw in xrange(num_waves):
            # s_index[0,:] = s[mw,:]
            for i in range(s_index.shape[1]):
                s_index[0, i] = s[mw,i]

            # amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            # apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]
            for j in range(wave.shape[2]):
                min_val = max_val = s_index[0, j]
                for k in range(1, s_index.shape[0]):
                     min_val = min(min_val, s_index[k,j])
                     max_val = max(max_val, s_index[k,j])

                amdq[m,j] += min_val * wave[m,mw,j]
                apdq[m,j] += max_val * wave[m,mw,j]

    return amdq, apdq

def godunov_update(num_rp, num_eqn, num_waves, wave, s):
    if use_numba: return godunov_update_compiled(num_rp, num_eqn, num_waves, wave, s)

    # Godunov update
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )
    s_index = np.zeros((2,num_rp))
    for m in xrange(num_eqn):
        for mw in xrange(num_waves):
            s_index[0,:] = s[mw,:]
            amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]

    return amdq, apdq

def compute_waves(a, a1, a2, a3, enthalpy, u, num_waves, num_rp):
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )

    wave[0,0,:] = a1
    wave[1,0,:] = a1 * (u-a)
    wave[2,0,:] = a1 * (enthalpy - u*a)
    s[0,:] = u - a

    wave[0,1,:] = a2
    wave[1,1,:] = a2 * u
    wave[2,1,:] = a2 * 0.5 * u**2
    s[1,:] = u

    wave[0,2,:] = a3
    wave[1,2,:] = a3 * (u+a)
    wave[2,2,:] = a3 * (enthalpy + u*a)
    s[2,:] = u + a

    return wave, s

def compute_eigenvector_coeff(q_l, q_r, a, u, enthalpy, gamma1):
    delta = q_r - q_l

    a2 = gamma1 / a**2 * ((enthalpy -u**2)*delta[0,:] + u*delta[1,:] - delta[2,:])
    a3 = (delta[1,:] + (a-u) * delta[0,:] - a*a2) / (2.0*a)
    a1 = delta[0,:] - a2 - a3

    return a1, a2, a3

def euler_hll_1D(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    HLL euler solver ::
    
         
        W_1 = Q_hat - Q_l    s_1 = min(u_l-c_l,u_l+c_l,lambda_roe_1,lambda_roe_2)
        W_2 = Q_r - Q_hat    s_2 = max(u_r-c_r,u_r+c_r,lambda_roe_1,lambda_roe_2)
    
        Q_hat = ( f(q_r) - f(q_l) - s_2 * q_r + s_1 * q_l ) / (s_1 - s_2)
    
    *problem_data* should contain:
     - *gamma* - (float) Ratio of the heat capacities
     - *gamma1* - (float) :math:`1 - \gamma`

    :Version: 1.0 (2014-03-04)
    """

    # Problem dimensions
    num_rp = q_l.shape[1]
    num_waves = 2

    # Return values
    wave = np.empty( (num_eqn, num_waves, num_rp) )
    s = np.empty( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )
    
    # Solver parameters
    gamma1 = problem_data['gamma1']
    
    # Calculate Roe averages, right and left speeds
    u, a, _, pl, pr = roe_averages(q_l,q_r,problem_data)
    H_r = (q_r[2,:] + pr) / q_r[0,:]
    H_l = (q_l[2,:] + pl) / q_l[0,:]
    u_r = q_r[1,:] / q_r[0,:]
    u_l = q_l[1,:] / q_l[0,:]
    a_r = np.sqrt(gamma1 * (H_r - 0.5 * u_r**2))
    a_l = np.sqrt(gamma1 * (H_l - 0.5 * u_l**2))

    # Compute Einfeldt speeds
    s_index = np.empty((4,num_rp))
    s_index[0,:] = u + a
    s_index[1,:] = u - a
    s_index[2,:] = u_l + a_l
    s_index[3,:] = u_l - a_l
    s[0,:]  = np.min(s_index,axis=0)
    s_index[2,:] = u_r + a_r
    s_index[3,:] = u_r - a_r
    s[1,:] = np.max(s_index,axis=0)

    # Compute middle state
    q_hat = np.empty((num_eqn,num_rp))
    q_hat[0,:] = (q_r[1,:] - q_l[1,:] 
                    - s[1,:] * q_r[0,:] + s[0,:] * q_l[0,:]) / (s[0,:] - s[1,:])
    q_hat[1,:] = (q_r[1,:]**2/q_r[0,:] + pr - (q_l[1,:]**2/q_l[0,:] + pl)
                    - s[1,:] * q_r[1,:] + s[0,:] * q_l[1,:]) / (s[0,:] - s[1,:])
    q_hat[2,:] = ((q_r[2,:] + pr)*q_r[1,:]/q_r[0,:] - (q_l[2,:] + pl)*q_l[1,:]/q_l[0,:] 
                    - s[1,:] * q_r[2,:] + s[0,:] * q_l[2,:]) / (s[0,:] - s[1,:])

    # Compute each family of waves
    wave[:,0,:] = q_hat - q_l
    wave[:,1,:] = q_r - q_hat
    
    # Compute variations
    s_index = np.zeros((2,num_rp))
    for m in xrange(num_eqn):
        for mw in xrange(num_waves):
            s_index[0,:] = s[mw,:]
            amdq[m,:] += np.min(s_index,axis=0) * wave[m,mw,:]
            apdq[m,:] += np.max(s_index,axis=0) * wave[m,mw,:]
            
    return wave, s, amdq, apdq

def euler_exact_1D(q_l,q_r,aux_l,aux_r,problem_data):
    r"""
    Exact euler Riemann solver
    
    .. warning::
        This solver has not been implemented.
    
    """
    raise NotImplementedError("The exact Riemann solver has not been implemented.")


def roe_averages(q_l,q_r,problem_data):
    return roe_averages(q_l, q_r, problem_data['gamma1'])

@numba.jit(nopython=True, cache=True)
def roe_averages_compiled(q_l, q_r, gamma1):
    #The current Numba version doesn't support non contiguous layouts 
    q_l0 = np.empty(q_l.shape[1], dtype=np.float64)
    q_l1 = np.empty(q_l.shape[1], dtype=np.float64)
    q_l2 = np.empty(q_l.shape[1], dtype=np.float64)
    pl = np.empty(q_l.shape[1], dtype=np.float64)
    rhsqrtl = np.empty(q_l.shape[1], dtype=np.float64)
    rhsqrtr = np.empty(q_l.shape[1], dtype=np.float64)
    rhsq2 = np.empty(q_l.shape[1], dtype=np.float64)
    u = np.empty(q_l.shape[1], dtype=np.float64)
    enthalpy = np.empty(q_l.shape[1], dtype=np.float64)
    a = np.empty(q_l.shape[1], dtype=np.float64)

    q_r0 = np.empty(q_r.shape[1], dtype=np.float64)
    q_r1 = np.empty(q_r.shape[1], dtype=np.float64)
    q_r2 = np.empty(q_r.shape[1], dtype=np.float64)
    pr = np.empty(q_r.shape[1], dtype=np.float64)

    for i in xrange(q_l.shape[1]):
        q_l0[i] = q_l[0,i]
        q_l1[i] = q_l[1,i]
        q_l2[i] = q_l[2,i]

        q_r0[i] = q_r[0,i]
        q_r1[i] = q_r[1,i]
        q_r2[i] = q_r[2,i]

        rhsqrtl[i] = sqrt(q_l0[i])
        rhsqrtr[i] = sqrt(q_r0[i])
        rhsq2[i] = rhsqrtl[i] + rhsqrtr[i]
        pl[i] = gamma1 * (q_l2[i] - 0.5 * (q_l1[i] ** 2) / q_l0[i])
        pr[i] = gamma1 * (q_r2[i] - 0.5 * (q_r1[i] ** 2) / q_r0[i])
        u[i] = (q_l1[i] / rhsqrtl[i] + q_r1[i] / rhsqrtr[i]) / rhsq2[i]
        enthalpy[i] = ((q_l2[i] + pl[i]) / rhsqrtl[i] + (q_r2[i] + pr[i]) / rhsqrtr[i]) / rhsq2[i]
        a[i] = sqrt(gamma1 * (enthalpy[i] - 0.5 * u[i] ** 2))

    return u, a, enthalpy, pl, pr

def roe_averages(q_l, q_r, gamma1):
    if use_numba: return roe_averages_compiled(q_l, q_r, gamma1)

    # Calculate Roe averages 
    rhsqrtl = np.sqrt(q_l[0,...])
    rhsqrtr = np.sqrt(q_r[0,...])
    pl = gamma1 * (q_l[2,...] - 0.5 * (q_l[1,...]**2) / q_l[0,...])
    pr = gamma1 * (q_r[2,...] - 0.5 * (q_r[1,...]**2) / q_r[0,...])
    rhsq2 = rhsqrtl + rhsqrtr
    u = (q_l[1,...] / rhsqrtl + q_r[1,...] / rhsqrtr) / rhsq2
    enthalpy = ((q_l[2,...] + pl) / rhsqrtl + (q_r[2,...] + pr) / rhsqrtr) / rhsq2
    a = np.sqrt(gamma1 * (enthalpy - 0.5 * u**2))

    return u, a, enthalpy, pl, pr