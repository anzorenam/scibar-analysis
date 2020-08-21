#!/usr/bin/env python3.7
# -*- coding: utf8 -*-

import numpy as np
import scipy.signal as signal
import scipy.integrate as integral
import os

home=os.environ['HOME']
dir='proyectos/scicrt/scibar-fitting'
name='{0}/{1}/19aug-7phe.adc'.format(home,dir)
Fs=2e9
f_mv=1000.0
f_ns=1e9
echarg=1.602e-7
# ganancia a -900V
#mu,sigma=0.607383, 0.120924
# ganancia a -950V
#mu,sigma=0.938888,0.146729
# ganancia a -980V es mayor
#mu,sigma=1.2318965391474737,0.2098413345370848
mu=1.2318965391474737+2.0*0.2098413345370848
#G=4993757.8 # usando LED azul
#G=5860724.1 # usando LED verde
G_fix=mu
x=-1.0*np.genfromtxt(name,dtype=np.float,delimiter=None)
maxmax=np.amax(x)
x=x[np.all(x<maxmax,axis=1)]
minmax=10.0e-3
test=np.any(x>minmax,axis=1)
x=x[test]
N=np.size(x,1)
M=np.size(x,0)
print(M,N)
dlist=np.ones(M,dtype=np.uint8)

ripp=20*np.log10(0.01)
bwidth=0.1
Ford,beta=signal.kaiserord(ripp,bwidth)
b=signal.firwin(Ford,0.15,window=('kaiser',beta))
y=signal.lfilter(b,1,x,axis=1)

for j in range(0,M):
  baseline=y[j,0:100]
  if np.all(baseline<0.005):
    f0=np.mean(baseline)
    y[j,:]-=f0
  else:
    dlist[j]=0

y=y[dlist!=0]
m0=f_mv*y
Ts=1.0/Fs
dt_norm=f_ns*Ts
t=f_ns*np.linspace(0,(N-1)/Fs,num=N)
m0de=(1.0/50.0)*integral.simps(m0,dx=dt_norm,axis=1,even='last')

m0de_norm=np.rint(m0de/(G_fix))
nout='{0}/{1}/19aug-7phe.hist'.format(home,dir)
np.savetxt(nout,m0de_norm,fmt='%3d',newline=' ')
