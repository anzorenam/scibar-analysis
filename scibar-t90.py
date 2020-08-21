#!/usr/bin/env python3.7
# -*- coding: utf8 -*-

import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import os

sns.set(rc={"figure.figsize":(8,4)})
sns.set_context('paper',font_scale=1.5,rc={'lines.linewidth':1.5})
sns.set_style('ticks')
mat.rc('text',usetex=True)
mat.rcParams['text.latex.preamble']=[r'\usepackage[utf8]{inputenc}',r'\usepackage[T1]{fontenc}',r'\usepackage[spanish]{babel}',r'\usepackage[scaled]{helvet}',r'\renewcommand\familydefault{\sfdefault}',r'\usepackage{amsmath,amsfonts,amssymb}',r'\usepackage{siunitx}']

home=os.environ['HOME']
dir='proyectos/scicrt/simulation/resultados-sim/pulse-shape'
nda0='{0}/{1}/19nov_2phe-t90.csv'.format(home,dir)
nda1='{0}/{1}/19aug_7phe-t90.csv'.format(home,dir)
nsim='{0}/{1}/ptimes_t9011.csv'.format(home,dir)

data0=np.loadtxt(nda0,dtype=np.float)
data1=np.loadtxt(nda1,dtype=np.float)
t90_exp=np.hstack((data0,data1))
#time_test=t90_exp>=100
#t90_exp=t90_exp[time_test]
t90_sim=np.loadtxt(nsim,dtype=np.float)
print(np.amax(t90_exp),np.amin(t90_exp))
print(np.amax(t90_sim),np.amin(t90_sim))
print(np.size(t90_exp,0),(1.0-np.size(t90_exp,0)/10220.0)*275.3)
tbins=np.arange(-200,300.0,0.1)
bins=np.arange(-100,230)
kde=stats.gaussian_kde(t90_exp,bw_method='silverman')
thist_exp=kde.evaluate(tbins)
kde=stats.gaussian_kde(t90_sim,bw_method='silverman')
thist_sim=kde.evaluate(tbins)
logic=np.logical_and(tbins<100.0,tbins>-200)
print(0.1*np.sum(thist_exp[logic]))
print(0.1*np.sum(thist_sim[logic])*256.9)
tm_exp=tbins[np.argmax(thist_exp)]
tm_sim=tbins[np.argmax(thist_sim)]
c=sns.color_palette(sns.cubehelix_palette(2,start=2,rot=0,dark=0,light=0.5,reverse=True))
sns.set_palette(c)
fig,ax=plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False)
ax.plot(tbins,thist_exp,color=c[0])
ax.hist(t90_exp,bins=bins,density=True,histtype='stepfilled',color=c[0])
ax.plot(tbins,thist_sim,color=c[1])
ax.hist(t90_sim,bins=bins,density=True,histtype='stepfilled',color=c[1])
plt.xlabel(r'$t_{90}-t_{10}$ $[\si{\nano\second}]$',x=0.9,horizontalalignment='right')
plt.ylabel(r'Probability density')
plt.xlim(0,160)
plt.tight_layout(pad=1.0)
plt.savefig('t90_dist.pdf')
plt.show()
