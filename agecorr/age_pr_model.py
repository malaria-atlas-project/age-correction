# Copyright (C) 2010 Anand Patil
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pylab as pl
import pymc as pm
from numpy import *

__all__ = ['make_model','make_MCMC']

def make_model(datasets):
    "Datasets should be a list of record arrays with columns [a_lo,a_hi,pos,neg]"

    def binom_deviance(n,y,p):
        y_hat = n*p
        return 2.*sum((y*log(y/y_hat))[where(y>0)]) + 2.*sum(((n-y) * log(n-y) / (n-y_hat))[where(y<n)])

    # c: .02-1.
    # alph: 18-30
    # b: .02-1.

    # alpha = array([10., 10., 10., 3.])
    sig_mean = array([.5, .4, .5, 1, .4, .5, 1])

    # sigma = pm.Gamma('sigma', alpha, alpha/sig_mean)
    sigma = pm.OneOverX('sigma', value=sig_mean)

    p_mean_mu = array([1.3, 3.7, 2.3, 0, 3.7, 2.3, 0])
    # p_mean = pm.MvNormal('p_mean', p_mean_mu, diag([10., 10., 10., 1.]))
    p_mean = pm.Uninformative('p_mean',value=p_mean_mu)

    R1 = pm.Uninformative('R1', zeros(6,dtype=float))
    R2 = pm.Uninformative('R2', zeros(3,dtype=float))
    R3 = pm.Uninformative('R3', zeros(3,dtype=float))

    @pm.deterministic
    def cholfac(R1=R1, R2=R2, R3=R3, sigma = sigma):
        """Cholesky factor of the covariance matrix."""

        cov = np.zeros((7,7),dtype=float)

        cov[0,0] = 1
        cov[1:,0] = R1
        cov[1:4,1:4] = pm.flib.expand_triangular(ones(3), R2)
        cov[4:7,4:7] = pm.flib.expand_triangular(ones(3), R3)    
    
        try:
            cholfac = asmatrix(diag(sigma)) * linalg.cholesky(cov)
        except linalg.LinAlgError:
            raise pm.ZeroProbability
        return cholfac
    
    Fs = []
    Ps= []
    data_list = []
    pred_data_list = []
    deviance_list = []
    pred_deviance_list = []
    p_vec_list = []
    P_prime_list = []
    for name in datasets.iterkeys():
    
        safe_name = name.replace('.','_')
    
        P_prime_now = pm.Beta('P_prime_%s'%safe_name,3.,3.)
        p_vec_now = pm.MvNormalChol('p_vec_%s'%safe_name, p_mean, cholfac)
        p_vec_list.append(p_vec_now)
        P_prime_list.append(P_prime_now)
    
        b = pm.Lambda('b', lambda p_vec = p_vec_now: 1./exp(p_vec[0]))
    
        if methods[name] == 'Microscopy':        
        
            # alpha, s and c depend on p_vec[1:4]
            c = pm.Lambda('c', lambda p_vec = p_vec_now: 1./exp(p_vec[1]))
            alph = pm.Lambda('alph', lambda p_vec = p_vec_now: exp(p_vec[2]))
            s = pm.Lambda('s', lambda p_vec = p_vec_now: pm.invlogit(p_vec[3]))
    
        elif methods[name] == 'RDT':
        
            # alpha, s and c depend on p_vec[4:7]
            c = pm.Lambda('c', lambda p_vec = p_vec_now: 1./exp(p_vec[4]))
            alph = pm.Lambda('alph', lambda p_vec = p_vec_now: exp(p_vec[5]))
            s = pm.Lambda('s', lambda p_vec = p_vec_now: pm.invlogit(p_vec[6]))        
        
        @pm.deterministic
        def this_F(c=c, alph=alph, a=age_bin_ctrs[name], s=s):
            """
            The function F, which gives detection probability.
            """
            out = empty(len(a))
            out[where(a<alph)] = 1.
            where_greater = where(a>=alph)
            out[where_greater] = (1.-s*(1.-exp(-c*(a-alph))))[where_greater]
            return out
        this_F.__name__ = 'F_%s'%safe_name
        
        @pm.deterministic
        def this_P(P_prime=P_prime_now, b=b, a=age_bin_ctrs[name], F=this_F):
            """
            The function P, which gives probability of a detected infection.
            """
            return P_prime * (1.-exp(-b*a)) * F
        this_P.__name__ = 'P_%s'%safe_name
    
        Fs.append(this_F)
        Ps.append(this_P)

        #Data
        this_data = pm.Binomial('data_%s'%safe_name, n=datasets[name].N, p=this_P, value=datasets[name].pos, isdata=True)
        data_list.append(this_data)
    
        # Initialize chain at MAP estimates to shorten burnin.
        M_start = pm.MAP([p_vec_now,P_prime_now,this_data])
        M_start.fit()

    # Samples from predictive distribution of parameters.
    p_pred = pm.MvNormalChol('p_predictive',p_mean,cholfac)
    b_pred = pm.Lambda('b', lambda p_vec = p_pred: 1./exp(p_vec[0]))
    c_pred = pm.Lambda('c', lambda p_vec = p_pred: 1./exp(p_vec[1]))
    alph_pred = pm.Lambda('alph', lambda p_vec = p_pred: exp(p_vec[2]))
    s_pred = pm.Lambda('s', lambda p_vec = p_pred: pm.invlogit(p_vec[3]))

    @pm.deterministic
    def F_pred(c=c_pred, alph=alph_pred, a=a, s=s_pred):
        """
        A sample from the predictive distribution of F.
        """
        out = empty(a.shape[0])
        out[where(a<alph)] = 1.
        where_greater = where(a>=alph)
        out[where_greater] = 1.-s*(1.-exp(-c*(a[where_greater]-alph)))
        return out

    @pm.deterministic
    def P_pred(P_prime=1., b=b_pred, a=a, F=F_pred):
        """
        A sample from the predictive distribuiton of P.
        """
        return P_prime * (1.-exp(-b*a)) * F
    
    return locals()


def make_MCMC(datasets, dbname):
    M = pm.MCMC(make_model(datasets), dbname=dbname, db='hdf5', dbcomplevel=1, dbcomplib='zlip')
    M.use_step_method(pm.AdaptiveMetropolis, [M.p_mean, M.R1, M.R2, M.R3, M.sigma], 
        scales={M.p_mean: .01*ones(7), M.R1: .01*ones(6), M.R2: .01*ones(3), M.R3: .01*ones(3), M.sigma: .01*ones(7)})

    for i in xrange(len(datasets)):
        M.use_step_method(pm.AdaptiveMetropolis, [M.p_vec_list[i],M.P_prime_list[i]], scales={M.p_vec_list[i]: .001*ones(7), M.P_prime_list[i]: [.001]}, delay=10000)