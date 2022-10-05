import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 600
mpl.rcParams["axes.axisbelow"] = False
import aa2022_figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
import pyreadr

pit = pyreadr.read_r("pit.RData")
pit = np.array(pit["pit"])

fig,ax = plt.subplots()
ax.hist(pit,bins=np.arange(0,1.1,0.1),color=u"#F4E87C",edgecolor=u"#FFDD33",lw=1,alpha=0.8)
ax.set_xlim(0,1)
ax.set_ylabel("Count")
ax.set_xlabel("Predictive integral transform value")
ax.text(0.09,17,"mean = "+str(np.round(np.mean(pit),3))+" (vs. 0.500)",backgroundcolor="white")
ax.text(0.09,10,"stdev = "+str(np.round(np.sqrt(np.var(pit)),3))+" (vs. 0.289)",backgroundcolor="white")
fig.savefig("figs\\pit.pdf", format='pdf', dpi=600, bbox_inches="tight")
fig.show()