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


import pymc as pm
from pymc.sandbox import GibbsStepMethods
from pymc.sandbox.GibbsStepMethods import DirichletMultinomial
from numpy import *
from age_pr_datasets import *

__all__ = ['make_model', 'make_MCMC']

def make_model(datasets):
    "Datasets should be a list of record arrays with columns [a,pos,neg]"

    N_pops = len(datasets)
    S_guess = exp(-a/20.)
    S_guess /= sum(S_guess)

    # Model for central tendency of age distribution.
    alph_pri = 10.

    scg = []
    for i in xrange(len(a)):
        asg_now = pm.Lambda('asg_%i'%i, lambda alpha=alph_pri, S=S_guess, i=i : alpha*S[i])
        scg.append(pm.Gamma('scg_%i'%i, alpha=asg_now, beta=1., value=asg_now.value))
    sc = pm.Lambda('sc',lambda gams=scg: array(gams)/sum(gams))

    # Model for age distributions of individual populations
    # alph = pm.Gamma('alph',2.,2./1000.)
    alph = pm.OneOverX('alph', value=50.)

    asc = pm.Lambda('asc',lambda alph=alph,S=sc: alph*S)

    S = []
    S_steps = []
    data_list = []
    for name in datasets.iterkeys():
    
        this_dataset = datasets[name]
    
        @pm.deterministic
        def this_asc_slice(asc=asc, slices = age_slices[name], dataset=this_dataset):
            out = []
            for i in xrange(len(dataset)):
                out.append(sum(asc[slices[i]]))
            out = array(out)
            return out
    
        S_now = pm.Dirichlet(('S_%s'%name).replace('.','_'), this_asc_slice)
        S.append(S_now)
        data_list.append(pm.Multinomial('data',n=sum(this_dataset.N),p=S_now,value=this_dataset.N,isdata=True))
    
    S_pred = pm.Dirichlet('S_pred',asc)
    
    return locals()

def make_mcmc(datasets, dbname)    
    M = pm.MCMC(make_model(datasets), db='hdf5',dbcomplevel=1, dbcomplib='zlib', dbname=dbname)

    M.use_step_method(pm.Metropolis, alph, sig=.05)

    for i in xrange(len(datasets)):    
        M.use_step_method(DirichletMultinomial, M.S[i])
