#!/usr/bin/env python3.7
# -*- coding: utf8 -*-

import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import root_numpy as rnp
import seaborn as sns
import scipy.stats as stats
import os

def resolution(x,R,k0,a1,a2):
  Kv=a1/np.power(1.0+a2*x,2.5)
  ed=np.arange(1,Nphe+1)
  sigma=R*np.sqrt(10.0*ed)
  y=np.zeros(Nphe)
  for j in range(1,500):
    gauss=stats.norm.pdf(x[:-1],loc=ed[j],scale=sigma[j])
    y[j]=np.sum(hsim*gauss)
  y=(k0/Kv)*y
  return y

sns.set(rc={"figure.figsize":(8,4)})
sns.set_context('paper',font_scale=1.5,rc={'lines.linewidth':1.5})
sns.set_style('ticks')
mat.rc('text',usetex=True)
mat.rcParams['text.latex.preamble']=[r'\usepackage[utf8]{inputenc}',r'\usepackage[T1]{fontenc}',r'\usepackage[spanish]{babel}',r'\usepackage[scaled]{helvet}',r'\renewcommand\familydefault{\sfdefault}',r'\usepackage{amsmath,amsfonts,amssymb}',r'\usepackage{siunitx}']

home=os.environ['HOME']
dir0='proyectos/scicrt/scibar-fitting'
dir1='proyectos/scicrt/simulation/resultados-sim/pulse-shape'
nda0='{0}/{1}/19nov-2phe.hist'.format(home,dir0)
nda1='{0}/{1}/19aug-7phe.hist'.format(home,dir0)
nsim='{0}/{1}/scibar_edep-s15.csv'.format(home,dir1)
data0=np.loadtxt(nda0,dtype=np.float)
data1=np.loadtxt(nda1,dtype=np.float)
dsim=np.loadtxt(nsim,dtype=np.float)
nphe_exp=np.hstack((data0,data1))

Nphe=500
qebins=np.arange(0,Nphe)
nphe_sim=dsim[:,4]
test=nphe_sim!=0.0
nphe_sim=nphe_sim[test]
Hnorm=np.size(nphe_sim)/np.size(nphe_exp)
w=False
hexp,b=np.histogram(nphe_exp,bins=qebins)
hsim,b=np.histogram(nphe_sim,bins=qebins)
kv=hexp*np.ma.divide(1.0,hsim).filled(0)
if w==True:
  np.savetxt('conv_fitting/sim_hist.dat',hsim)
  np.savetxt('conv_fitting/exp_hist.dat',hexp)
  np.savetxt('conv_fitting/phe_sat.dat',kv)

conv=resolution(qebins,1.02762,0.0174426,0.0569227,0.0178782)
conv[5]=0.6*conv[5]
conv[6]=0.7*conv[6]
c=sns.cubehelix_palette(8, start=2,rot=0,dark=0,light=.95,reverse=True)
fig,ax=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=False)
ax.plot(qebins[:-1],hsim,ds='steps-mid',color=c[0])
ax.plot(qebins[:-1],hexp,ds='steps-mid',color=c[2])
ax.plot(qebins[3:140],conv[3:140],ds='steps-mid',ls=':',lw=2.5,color=c[1])
plt.yscale('log')
plt.xscale('log')
plt.ylim(1e0,1e4)
plt.xlim(5,250)
plt.xlabel(r'Photoelectrons',x=0.95,horizontalalignment='right')
plt.ylabel(r'$\log_{10}\left(\text{Counts}\right)$')
plt.tight_layout(pad=1.0)
plt.savefig('photons-number.pdf')
plt.show()
