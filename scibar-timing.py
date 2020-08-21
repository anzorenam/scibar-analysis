#!/usr/bin/env python3.7
# -*- coding: utf8 -*-

import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import scipy.integrate as integral
import ROOT
import root_numpy as rnp
import seaborn as sns
import os
import mpl_toolkits.axes_grid1.inset_locator as mpl

sns.set(rc={"figure.figsize":(8,4)})
sns.set_context('paper',font_scale=1.5,rc={'lines.linewidth':1.5})
sns.set_style('ticks')
mat.rc('text',usetex=True)
mat.rcParams['text.latex.preamble']=[r'\usepackage[utf8]{inputenc}',r'\usepackage[T1]{fontenc}',r'\usepackage[spanish]{babel}',r'\usepackage[scaled]{helvet}',r'\renewcommand\familydefault{\sfdefault}',r'\usepackage{amsmath,amsfonts,amssymb}',r'\usepackage{siunitx}']

home=os.environ['HOME']
name='{0}/proyectos/scicrt/scibar-fitting/19nov-2phe.adc'.format(home)
Fs=4e9
delay=60.8
#delay=175.2
f_mv=1000.0
f_ns=1e9

x=np.genfromtxt(name,dtype=np.float,delimiter=None)
print(np.size(x,0))
maxmax=np.amax(x)
x=x[np.all(x<maxmax,axis=1)]
minmax=20.0e-3
test=np.any(x>minmax,axis=1)
x=x[test]
N=np.size(x,1)
M=np.size(x,0)
print(M,N)
dlist=np.ones(M,dtype=np.uint8)

ripp=20*np.log10(0.01)
bwidth=0.1
Ford,beta=signal.kaiserord(ripp,bwidth)
b=signal.firwin(Ford,0.1,window=('kaiser',beta))
y=signal.lfilter(b,1,x,axis=1)

for j in range(0,M):
  baseline=y[j,0:100]
  if np.all(baseline<0.005):
    f0=np.mean(baseline)
    y[j,:]-=f0
  else:
    dlist[j]=0

y=y[dlist!=0]
Mtot=np.size(y,0)
m0=f_mv*y
Ts=1.0/Fs
dt_norm=f_ns*Ts
t=f_ns*np.linspace(0,(N-1)/Fs,num=N)
tbins=np.arange(-100,500.0,1.0)
m0max=1.0/np.amax(m0,axis=1)
m0norm=m0*np.transpose(m0max[np.newaxis])
m01=1.0/integral.trapz(m0,dx=dt_norm,axis=1)
m0n=m0*np.transpose(m01[np.newaxis])

dt0=integral.cumtrapz(m0n,dx=dt_norm,axis=1,initial=0)
t90=dt_norm*np.sum(np.logical_and(dt0<=0.9,dt0>=0.1),axis=1)

ptimes=np.argmax(m0norm,axis=1)
tmax=t[ptimes]-delay
dir='proyectos/scicrt/simulation/resultados-sim/pulse-shape'
ntmax='{0}/{1}/19nov_2phe-tmax.csv'.format(home,dir)
np.savetxt(ntmax,tmax,newline=' ')

tdist=ROOT.TF1('tmuon','expo',10,40)

Nwin=400
tgraph=ROOT.TGraph()
m0mean=np.zeros(Nwin)
m0std=np.zeros(Nwin)
ptimes=np.argmax(m0norm,axis=1)
tau0=1.0
tau1=17.241
jtot=0

for j in range(0,Mtot):
  if (ptimes[j]+Nwin)<=N:
    t0=t[ptimes[j]]
    m0pulse=m0norm[j,ptimes[j]:ptimes[j]+Nwin]
    tmax=t0+1.0*np.log(tau1/(tau0))*(tau0*tau1/(tau0-tau1))
    texp=t-tmax
    u=t>tmax
    pulse_mod=(np.exp(-1.0*texp/tau1)-np.exp(-1.0*texp/tau0))*u
    m0mean+=m0pulse
    #kmod=np.amax(pulse_mod)
    #wdiff=np.cumsum(-vphe,axis=1)
    m0std+=np.power(m0pulse,2.0)
    jtot+=1

kmax=1.0/np.amax(m0mean)
m0std=(1.0/Mtot)*(m0std-np.power(m0mean,2.0))
m0mean=m0mean*kmax
m0std=m0std*kmax
timeT=np.transpose(np.array([t[0:Nwin],m0mean]))
rnp.fill_graph(tgraph,timeT)
tgraph.Fit(tdist,'R','Q')
tpars=tdist.GetParameters()
mexp_exp=np.exp(tpars[0]+1.0*t*tpars[1])
print(1.0/tpars[1],jtot)
print(np.amax(t90),np.amin(t90))

dir='proyectos/scicrt/simulation/resultados-sim/pulse-shape'
ndense='{0}/{1}/tail-fit_e11.csv'.format(home,dir)
dense_tot=np.loadtxt(ndense)
tbins=np.arange(0,800,0.5)
tmax_sim=39.5
tdist=ROOT.TF1('tmuon','expo',50,100)
tgraph=ROOT.TGraph()

timeT=np.transpose(np.array([tbins,dense_tot]))
rnp.fill_graph(tgraph,timeT)

tgraph.Fit(tdist,'R')
tpars=tdist.GetParameters()
mexp_sim=np.exp(tpars[0]+1.0*tbins*tpars[1])
print(1.0/tpars[1])

c=sns.color_palette(sns.cubehelix_palette(2,dark=0.15,light=.5,reverse=True))
sns.set_palette(c)
fig,ax=plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False)
ax.errorbar(t[0:Nwin],m0mean,yerr=m0std,errorevery=4,color=c[0])
ax.semilogy(t,mexp_exp,ls=':',color=c[0])
plt.xlabel(r'Time $[\si{\nano\second}]$',x=0.9,horizontalalignment='right')
plt.ylabel(r'Amplitude $[\si{normalized}]$')
plt.xlim(0,100)
plt.ylim(1e-2,1e1)
ax_inset=mpl.inset_axes(ax,width='40%',height='40%',loc=1)
ax_inset.semilogy(tbins-tmax_sim,dense_tot,color=c[0])
ax_inset.semilogy(tbins-tmax_sim,mexp_sim,ls='--',color=c[0])
plt.xlabel(r'Time $[\si{\nano\second}]$',x=0.9,horizontalalignment='right')
plt.ylabel(r'Amplitude')
plt.xlim(0,99)
plt.ylim(1e-2,1e1)
plt.savefig('muons-tail-fit.pdf',bbox_inches='tight')
