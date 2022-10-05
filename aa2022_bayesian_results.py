import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 600
mpl.rcParams["axes.axisbelow"] = False
import aa2022_figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
from scipy.stats import gamma
from scipy.stats import norm
from scipy import optimize
from scipy import interpolate 
import pyreadr
from aa2022_read_ephemeris import load_ephemeris
import datetime as dt
from aa2022_conversions import date2jd
from aa2022_conversions import jd2date
import pandas as pd
from aa2022_download import fetch
from scipy.signal import savgol_filter
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
import mpl_toolkits.axisartist as aa
from mpl_toolkits.axes_grid1 import make_axes_locatable

au = 149597870.7 #astronomical unit, km

def hpd(vector,percentile=90):
    #only works with unimodal, but good enough for us here
    hst = np.histogram(vector,bins=int((len(vector)**0.333)))
    pdf = interpolate.interp1d((hst[1][:-1]+hst[1][1:])/2,hst[0],fill_value="extrapolate",kind=3)
    vec_pdf = pdf(vector)

    inds = (-vec_pdf).argsort()
    sorted_vector = vector[inds]
    
    mass_length = int(np.ceil(len(vector)*percentile/100))
       
    return [np.min(sorted_vector[:mass_length]),np.max(sorted_vector[:mass_length])]

#%% log plot of priors + posteriors

#priors
b1_x = np.arange(1.4,5.0,0.01)
b1_y = norm.pdf(b1_x,2.2,scale=(0.2))
#b1_y = norm.pdf(b1_x,2.2,scale=(0.5))         #corrected_vague
#b1_y = norm.pdf(b1_x,1.9,scale=(0.2))         #new_to_left

b2_x = np.arange(-3,-0.5,0.01)
b2_y = norm.pdf(b2_x,-1.8,scale=(0.2))
#b2_y = norm.pdf(b2_x,-1.8,scale=(0.5))        #corrected_vague 
#b2_y = norm.pdf(b2_x,-2.1,scale=(0.2))        #new_to_left

c1_x = np.arange(0,11,0.01)
c1_y = gamma.pdf(c1_x,3,scale=1)
#c1_y = gamma.pdf(c1_x,2,scale=1.5)   #corrected_vague
#c1_y = gamma.pdf(c1_x,3,scale=0.5)   #new_to_left

c2_x = np.arange(0,11,0.01)
c2_y = gamma.pdf(c2_x,3,scale=1)
#c2_y = gamma.pdf(c2_x,2,scale=1.5)   #corrected_vague
#c2_y = gamma.pdf(c2_x,3,scale=0.5)   #new_to_left

v1_x = np.arange(0,110,0.1)
v1_y = gamma.pdf(v1_x,10,scale=5)
#v1_y = gamma.pdf(v1_x,5,scale=10)         #corrected_vague
#v1_y = gamma.pdf(v1_x,10,scale=2.5)       #new_to_left

#priors new champion
xb1_x = np.arange(1.4,5.0,0.01)
xb1_y = norm.pdf(b1_x,2.2,scale=(0.2))
xb2_x = np.arange(-3,-0.5,0.01)
xb2_y = norm.pdf(b2_x,-1.8,scale=(0.2))
xc1_x = np.arange(0,11,0.01)
xc1_y = gamma.pdf(c1_x,3,scale=1)
xc2_x = np.arange(0,11,0.01)
xc2_y = gamma.pdf(c2_x,3,scale=1)
xv1_x = np.arange(0,110,0.1)
xv1_y = gamma.pdf(v1_x,10,scale=5)

scale = 1
prior_color = u"#af1b3f"
posterior_color = u"#fde74c"
mean_color = u"#fde74c"
filename = "champion.RData" #"sample.RData" #
overplot_prior_champion = True
overplot_posterior_champion = True
if overplot_posterior_champion:
    posterior_color = u'#623CEA' 
    mean_color = u'#623CEA'
if overplot_prior_champion:    
    prior_color = u"#31A97F"

bins = 30
fig = plt.figure(figsize=(3,4))
gs = fig.add_gridspec(5, hspace=.6)
axs = gs.subplots()

if overplot_prior_champion:
    axs[0].plot(xb1_x, scale*xb1_y,color=u"#af1b3f",ls="dashed",lw=1)  
    axs[1].plot(xb2_x, scale*xb2_y,color=u"#af1b3f",ls="dashed",lw=1)
    axs[2].plot(xc1_x, scale*xc1_y,color=u"#af1b3f",ls="dashed",lw=1)
    axs[3].plot(xc2_x, scale*xc2_y,color=u"#af1b3f",ls="dashed",lw=1,label="Ref. prior")
    axs[4].plot(xv1_x, scale*xv1_y,color=u"#af1b3f",ls="dashed",lw=1)

axs[0].fill_between(b1_x, scale*b1_y,lw=0,color=prior_color,alpha=0.5)  
axs[0].set_xlabel(r"$\epsilon_v$")
axs[0].set_xlim(left=1.4, right=3.1)
axs[1].fill_between(b2_x, scale*b2_y,lw=0,color=prior_color,alpha=0.5)
axs[1].set_xlabel(r"$\epsilon_r$")
axs[1].set_xlim(left=-2.59, right=-0.91)
axs[2].fill_between(c1_x, scale*c1_y,lw=0,color=prior_color,alpha=0.5,label="Prior")
axs[2].set_xlabel(r"$\lambda_\beta$")
axs[2].set_xlim(left=-0.5, right=10.5)
axs[3].fill_between(c2_x, scale*c2_y,lw=0,color=prior_color,alpha=0.5)
axs[3].set_xlabel(r"$\lambda_{bg}$")
axs[3].set_xlim(left=-0.5, right=10.5)
axs[4].fill_between(v1_x, scale*v1_y,lw=0,color=prior_color,alpha=0.5)
axs[4].set_xlabel(r"$\nu_r$")
axs[4].set_xlim(left=-5, right=105)


#posteriors
samples = pyreadr.read_r(filename)
b1 = np.array(samples["b1"])
b2 = np.array(samples["b2"])
c1 = np.array(samples["c1"])
c2 = np.array(samples["c2"])
v1 = np.array(samples["v1"])

hst_b1 = np.histogram(b1,bins=bins,density=True)
hst_b2 = np.histogram(b2,bins=bins,density=True)
hst_c1 = np.histogram(c1[c1<5],bins=bins,density=True)
hst_c2 = np.histogram(c2[c2<5],bins=bins,density=True)
hst_v1 = np.histogram(v1[(v1<110)*(v1>30)],bins=bins,density=True)

axs[0].fill_between((hst_b1[1][1:]+hst_b1[1][:-1])/2,hst_b1[0],lw=0,color=posterior_color,alpha=0.5)
axs[1].fill_between((hst_b2[1][1:]+hst_b2[1][:-1])/2,hst_b2[0],lw=0,color=posterior_color,alpha=0.5)
axs[2].fill_between((hst_c1[1][1:]+hst_c1[1][:-1])/2,hst_c1[0],lw=0,color=posterior_color,alpha=0.5,label="Posterior")
axs[3].fill_between((hst_c2[1][1:]+hst_c2[1][:-1])/2,hst_c2[0],lw=0,color=posterior_color,alpha=0.5)
axs[4].fill_between((hst_v1[1][1:]+hst_v1[1][:-1])/2,hst_v1[0],lw=0,color=posterior_color,alpha=0.5)

axs[0].vlines(np.mean(b1),ymin=0,ymax=np.max(hst_b1[0]),color=mean_color,lw=1)
axs[1].vlines(np.mean(b2),ymin=0,ymax=np.max(hst_b2[0]),color=mean_color,lw=1)
axs[2].vlines(np.mean(c1),ymin=0,ymax=np.max(hst_c1[0]),color=mean_color,lw=1)
axs[3].vlines(np.mean(c2),ymin=0,ymax=np.max(hst_c2[0]),color=mean_color,lw=1)
axs[4].vlines(np.mean(v1),ymin=0,ymax=np.max(hst_v1[0]),color=mean_color,lw=1)

if overplot_posterior_champion:
    samples = pyreadr.read_r("champion.RData")
    b1 = np.array(samples["b1"])
    b2 = np.array(samples["b2"])
    c1 = np.array(samples["c1"])
    c2 = np.array(samples["c2"])
    v1 = np.array(samples["v1"])
    
    hst_b1 = np.histogram(b1,bins=bins,density=True)
    hst_b2 = np.histogram(b2,bins=bins,density=True)
    hst_c1 = np.histogram(c1,bins=bins,density=True)
    hst_c2 = np.histogram(c2,bins=bins,density=True)
    hst_v1 = np.histogram(v1,bins=bins,density=True)
    
    axs[0].plot((hst_b1[1][1:]+hst_b1[1][:-1])/2,hst_b1[0],ls="dashed",color=u"#fde74c",lw=1)
    axs[1].plot((hst_b2[1][1:]+hst_b2[1][:-1])/2,hst_b2[0],ls="dashed",color=u"#fde74c",lw=1)
    axs[2].plot((hst_c1[1][1:]+hst_c1[1][:-1])/2,hst_c1[0],ls="dashed",color=u"#fde74c",lw=1)
    axs[3].plot((hst_c2[1][1:]+hst_c2[1][:-1])/2,hst_c2[0],ls="dashed",color=u"#fde74c",lw=1,label="Ref. post.")
    axs[4].plot((hst_v1[1][1:]+hst_v1[1][:-1])/2,hst_v1[0],ls="dashed",color=u"#fde74c",lw=1)
    
    axs[0].vlines(np.mean(b1),ymin=0,ymax=max(hst_b1[0]),color=u"#fde74c",ls="dashed",lw=1)
    axs[1].vlines(np.mean(b2),ymin=0,ymax=max(hst_b2[0]),color=u"#fde74c",ls="dashed",lw=1)
    axs[2].vlines(np.mean(c1),ymin=0,ymax=max(hst_c1[0]),color=u"#fde74c",ls="dashed",lw=1)
    axs[3].vlines(np.mean(c2),ymin=0,ymax=max(hst_c2[0]),color=u"#fde74c",ls="dashed",lw=1)
    axs[4].vlines(np.mean(v1),ymin=0,ymax=max(hst_v1[0]),color=u"#fde74c",ls="dashed",lw=1)

axs[2].legend(fontsize="x-small",frameon=True,edgecolor="none",framealpha=0.8,loc=[0.67,0.3])
axs[3].legend(fontsize="x-small",frameon=True,edgecolor="none",framealpha=0.8,loc=[0.67,0.3])

axs[0].set_ylim(bottom=0.005,top=8)
axs[1].set_ylim(bottom=0.005,top=8)
axs[2].set_ylim(bottom=0.005,top=8)
axs[3].set_ylim(bottom=0.005,top=8)
axs[4].set_ylim(bottom=0.000125,top=0.2)

for i in range(5):
    axs[i].tick_params(axis="x",which="both",top=False)
    axs[i].tick_params(axis="y",which="both",right=False)
    axs[i].tick_params(axis="y",which="minor",left=False)
    axs[i].xaxis.set_label_coords(0.5, -0.2)
    axs[i].yaxis.set_label_coords(-0.15, 0.5)
    axs[i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    axs[i].set_axisbelow(False)
    axs[i].tick_params(zorder=50)
    axs[i].set_yscale('log')
    
axs[2].set_ylabel("Probability density")

fig.savefig("figs\\priors_posteriors.pdf", format='pdf', dpi=600, bbox_inches="tight")
fig.show()


#%% estimating origin and beta value from samples of b2, v1

au = 149597870.7 #astronomical unit, km
GM = 1.327e20 #gravity of Sun, m3 s-2   
sample_size = 1000

def orbital_velocity(a): #a in AU, result in km/s
    return np.sqrt(GM/(a*au*1000))/1000

def beta_velocity_radial(a,r,beta=0):
    # a is initial orbit
    # r is position
    # beta is beta-factor
    # radial motion is assumed
    v_total = np.sqrt( (orbital_velocity(a)*1000)**2 
                   + 2*GM*(1-beta)*(1/(r*au*1000)-1/(a*au*1000)) )/1000
    v_tangential = orbital_velocity(a) * a / r
    v_radial = np.sqrt(v_total**2 - v_tangential**2)
    return v_radial
v_beta_velocity_radial = np.vectorize(beta_velocity_radial)
    
def a_estimate(beta,v1):
    kappa = GM*(1-beta)  #reguced GM
    v_tot_sq = (v1*1000)**2 + (12*1000)**2   #in m/s
    a = (GM - 2*kappa) / (v_tot_sq - 2*kappa/(0.75*au*1000))
    return a/(1000*au) #in au

v_a_estimate = np.vectorize(a_estimate)

#print(a_estimate(0.7,60))

def b2_estimate(beta,a):  
    span = np.arange(0.5,1.05,0.05)
    dust_radial_velocity = v_beta_velocity_radial(a,span,beta=beta)
    synthetic_density = 100*(1/(span**2))*(1/dust_radial_velocity)
    def density_function(x,c,b):
        return c*x**(b)
    fitted = optimize.curve_fit(density_function, 
                                span, 
                                synthetic_density, 
                                p0=[10, -2])
    return fitted[0][1]  

#print(b1_estimate(0.6, 0.1))

def beta_estimate(ia,ib):
    #error = lambda x : (b1_estimate(x,ia) - ib)**2
    b2_fixed_a = lambda beta : b2_estimate(beta,ia)
    v_b2_fixed_a = np.vectorize(b2_fixed_a)
    betas = np.arange(0.5,1.01,0.01)
    b1s = v_b2_fixed_a(betas)
    errors = b1s - ib
    err = interpolate.interp1d(errors,betas,fill_value="extrapolate",kind=3)
    return err(0)

v_beta_estimate = np.vectorize(beta_estimate)
    
#print(beta_estimate(0.1,-1.9))

beta5 = v_beta_estimate(0.025,b2[:sample_size])
beta10 = v_beta_estimate(0.05,b2[:sample_size])
beta20 = v_beta_estimate(0.1,b2[:sample_size])
beta40 = v_beta_estimate(0.2,b2[:sample_size])

bins = np.arange(0.4,0.63,0.02)

hst_5  = np.histogram(beta5,bins=bins,density=True)
hst_10 = np.histogram(beta10,bins=bins,density=True)
hst_20 = np.histogram(beta20,bins=bins,density=True)
hst_40 = np.histogram(beta40,bins=bins,density=True)

df_5  = interpolate.interp1d((hst_5[1][1:] +hst_5[1][:-1])/2 ,hst_5[0], fill_value="extrapolate",kind=3)
df_10 = interpolate.interp1d((hst_10[1][1:]+hst_10[1][:-1])/2,hst_10[0],fill_value="extrapolate",kind=3)
df_20 = interpolate.interp1d((hst_20[1][1:]+hst_20[1][:-1])/2,hst_20[0],fill_value="extrapolate",kind=3)
df_40 = interpolate.interp1d((hst_40[1][1:]+hst_40[1][:-1])/2,hst_40[0],fill_value="extrapolate",kind=3)



fig,ax = plt.subplots()
span = np.arange(0.4,0.62,0.001)


ax.plot(span,df_10(span),lw=2,ls=(0,(4,2)),color=u"#ec9a29",alpha=0.7,label="Origin at $0.05 AU$")
ax.plot(span,df_20(span),lw=2,ls="solid",color=u"#a8201a",alpha=0.7,label="Origin at $0.1 AU$")
ax.plot(span,df_40(span),lw=2,ls=(0,(1,1)),color=u"#0f8b8d",alpha=0.7,label="Origin at $0.2 AU$")

ax.legend()
ax.set_ylim(0,30)
ax.set_xlim(0.401,0.609)
ax.set_aspect(0.208/(30*1.618))
ax.set_xlabel(r"$\beta$ parameter")
ax.set_ylabel(r"Probability density")
ax.legend(loc=1,fontsize="small",edgecolor="white")
#ax.yaxis.set_ticklabels([])
fig.tight_layout(pad=0.5)
fig.savefig("figs\\beta_parameter.pdf", format='pdf', dpi=600)
fig.show()    


    
#%% sampling full profile of flux and overplotting with actual data

# --------- plotting expected flux

sample_size_mu = 1
sample_size_value = 100


def mu(b1, b2, c1, c2, v1, r, vr, vt):
    return ( (((vr-v1)**2+(vt-(12*0.75/r))**2)**0.5)/50 )**(b1)*r**(b2)*c1 + c2

jd_ephem, hae_r, hae_v, hae_phi, radial_v, tangential_v = load_ephemeris("solo")
heliocentric_distance = np.sqrt(hae_r[:,0]**2+hae_r[:,1]**2+hae_r[:,2]**2)/au #in au
f_heliocentric_distance = interpolate.interp1d(jd_ephem,heliocentric_distance,fill_value="extrapolate",kind=3)
f_rad_v = interpolate.interp1d(jd_ephem,radial_v,fill_value="extrapolate",kind=3)
f_tan_v = interpolate.interp1d(jd_ephem,tangential_v,fill_value="extrapolate",kind=3)

jd_span = np.arange(2459028.5,2459572.5,14)
date_span = np.zeros(len(jd_span),dtype=dt.date)
mean = np.zeros(len(jd_span))
q5 = np.zeros(len(jd_span))
q25 = np.zeros(len(jd_span))
q75 = np.zeros(len(jd_span))
q95 = np.zeros(len(jd_span))

for i in range(len(jd_span)):
    jd = jd_span[i] 
    date_span[i] = jd2date(jd)
    r = f_heliocentric_distance(jd)
    vr = f_rad_v(jd)
    vt = f_tan_v(jd)
    rates = np.zeros(0)
    for j in range(sample_size_mu):
        ib1 = b1[j]
        ib2 = b2[j]
        ic1 = c1[j]
        ic2 = c2[j]
        iv1 = v1[j]
        imu = mu(ib1,ib2,ic1,ic2,iv1,r,vr,vt)
        for k in range(sample_size_value):
            exposure = 1.463 #hours
            rate = max(np.random.normal(imu,0),0)
            rates = np.append(rates,np.random.poisson(lam=rate*exposure,size=sample_size_value)/exposure)
    mean[i] = np.mean(rates)
    
    sample_day = rates+np.random.normal(0,0.2,size=len(rates))
    
    hpd50 = hpd(sample_day,percentile=50)
    hpd90 = hpd(sample_day,percentile=90)
    
    q5[i]  = hpd90[0]
    q95[i] = hpd90[1]
    q25[i] = hpd50[0]
    q75[i] = hpd50[1]
    

window = 3
order = 2

mean_s = savgol_filter(mean, window, order)
q5_s = savgol_filter(q5, window, order)
q25_s = savgol_filter(q25, window, order)
q75_s = savgol_filter(q75, window, order)
q95_s = savgol_filter(q95, window, order)

f_mean = interpolate.interp1d(jd_span, mean_s, fill_value="extrapolate", kind=3)
f_q5 = interpolate.interp1d(jd_span, q5_s, fill_value="extrapolate", kind=3)
f_q25 = interpolate.interp1d(jd_span, q25_s, fill_value="extrapolate", kind=3)
f_q75 = interpolate.interp1d(jd_span, q75_s, fill_value="extrapolate", kind=3)
f_q95 = interpolate.interp1d(jd_span, q95_s, fill_value="extrapolate", kind=3)

v_f_mean = np.vectorize(f_mean)
v_f_q5 = np.vectorize(f_q5)
v_f_q25 = np.vectorize(f_q25)
v_f_q75 = np.vectorize(f_q75)
v_f_q95 = np.vectorize(f_q95)

fine_jd_span = np.arange(2459028.5,2459572.5,1)
fine_date_span = np.zeros(len(fine_jd_span),dtype=dt.date)
for i in range(len(fine_jd_span)):
    jd = fine_jd_span[i] 
    fine_date_span[i] = jd2date(jd)
    
fine_mean = v_f_mean(fine_jd_span)
fine_q5 = v_f_q5(fine_jd_span)
fine_q25 = v_f_q25(fine_jd_span)
fine_q75 = v_f_q75(fine_jd_span)
fine_q95 = v_f_q95(fine_jd_span)


# --------- plotting detected flux

#read CNN csv into np arrays and disregard nans
dust_counts = pd.read_csv("cnn.txt",delim_whitespace=True)
flux = np.array(dust_counts["daily_dust_count"])
#maybe add stdev? maybe not
years = np.zeros(0,dtype=int)
months = np.zeros(0,dtype=int)
days = np.zeros(0,dtype=int)
time_scanned = np.zeros(0,dtype=int)

# correcting for exposure
for date in dust_counts["days"]:
    years = np.append(years,int(date[0:4]))
    months = np.append(months,int(date[5:7]))
    days = np.append(days,int(date[8:10]))
    YYYYMMDD = str(int(date[0:4]))+str(int(date[5:7])).zfill(2)+str(int(date[8:10])).zfill(2)
    try:
        online = False
        cdf_file = fetch(YYYYMMDD,'data',"tds_stat","_rpw-tds-surv-stat_",["V04.cdf","V03.cdf","V02.cdf","V01.cdf"],online)
        time_scanned = np.append(time_scanned,sum(cdf_file.varget("snapshot_len")*cdf_file.varget("SN_NR_EVENTS")/cdf_file.varget("sampling_rate")/3600))
        #time in hours = samples per window * number of windows / ( sampling rate per second * seconds / hour )
    except:
        time_scanned = np.append(time_scanned,np.nan)
        
mask = ~np.isnan(flux) * ~np.isnan(time_scanned)
flux = flux[mask]
    
jd_flux = np.zeros(len(flux))
for i in range(len(flux)):
    jd_flux[i] = date2jd(dt.date(years[mask][i],months[mask][i],days[mask][i]))

flux_date = np.zeros(len(jd_flux),dtype=dt.date)
for i in range(len(jd_flux)):
    flux_date[i] = jd2date(jd_flux[i]).date()

time_scanned_masked = time_scanned[mask]
flux_cnn = flux / (time_scanned_masked)

#%% plotting
fig,ax = plt.subplots()
ax.scatter(flux_date, flux_cnn*24, color = u"#F15025", s=1, zorder = 1000)#, label = "CNN impacts")
ax.plot(fine_date_span,fine_mean*24,color=u"#57467B",label="Mean")
ax.fill_between(fine_date_span, fine_q25*24, fine_q75*24, linewidth = 0, color = u"#9787ba", alpha = 1,label="50\%", zorder = 1)
ax.fill_between(fine_date_span, fine_q5*24, fine_q95*24, linewidth = 0, color = u"#57467B", alpha = 0.4,label="90\%", zorder = 0)

ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.set_ylim(0,520)
ax.set_xlim(np.min(fine_date_span),np.max(fine_date_span))
ax.tick_params(axis='x',which="minor",bottom=True,top=True)
ax.tick_params(axis='x',labelrotation=45 )
ax.legend(loc=(0.08,0.65),fontsize="small",frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax.set_ylabel(r"Impact rate [$day^{-1}$]")
ax.set_aspect((543)/(520*1.618))
ax2 = ax.twinx()
ax2.set_ylim(0,2.708333)
ax2.set_ylabel(r"Impact rate [$m^{-2} h^{-1}$]")
fig.tight_layout(pad=0.5)
fig.savefig('figs\\modelled_flux.pdf', format='pdf', dpi=600)
fig.show()

#count how well the intervals actually do
in50 = 0
in90 = 0
for point in range(len(flux_date)):
    jd = date2jd(flux_date[point])
    if v_f_q25(jd) < flux_cnn[point] < v_f_q75(jd):
        in50 += 1
    if v_f_q5(jd) < flux_cnn[point] < v_f_q95(jd):
        in90 += 1
print("50%: "+str(in50/len(flux_date)))
print("90%: "+str(in90/len(flux_date)))

#%% plotting detection rate uncertainty

# --------- plotting expected flux

sample_size_mu = 10000
sample_size_value = 1


def mu(b1, b2, c1, c2, v1, r, vr, vt):
    return ( (((vr-v1)**2+(vt-(12*0.75/r))**2)**0.5)/50 )**(b1)*r**(b2)*c1 + c2

jd_ephem, hae_r, hae_v, hae_phi, radial_v, tangential_v = load_ephemeris("solo")
heliocentric_distance = np.sqrt(hae_r[:,0]**2+hae_r[:,1]**2+hae_r[:,2]**2)/au #in au
f_heliocentric_distance = interpolate.interp1d(jd_ephem,heliocentric_distance,fill_value="extrapolate",kind=3)
f_rad_v = interpolate.interp1d(jd_ephem,radial_v,fill_value="extrapolate",kind=3)
f_tan_v = interpolate.interp1d(jd_ephem,tangential_v,fill_value="extrapolate",kind=3)

jd_span = np.arange(2459028.5,2459572.5,14)
date_span = np.zeros(len(jd_span),dtype=dt.date)
mean = np.zeros(len(jd_span))
q5 = np.zeros(len(jd_span))
q25 = np.zeros(len(jd_span))
q75 = np.zeros(len(jd_span))
q95 = np.zeros(len(jd_span))

for i in range(len(jd_span)):
    jd = jd_span[i] 
    date_span[i] = jd2date(jd)
    r = f_heliocentric_distance(jd)
    vr = f_rad_v(jd)
    vt = f_tan_v(jd)
    rates = np.zeros(0)
    for j in range(sample_size_mu):
        ib1 = b1[j]
        ib2 = b2[j]
        ic1 = c1[j]
        ic2 = c2[j]
        iv1 = v1[j]
        imu = mu(ib1,ib2,ic1,ic2,iv1,r,vr,vt)
        for k in range(sample_size_value):
            exposure = 1.463 #hours
            rate = max(np.random.normal(imu,0),0)
            rates = np.append(rates,rate)
    mean[i] = np.mean(rates)
    
    sample_day = rates+np.random.normal(0,0.2,size=len(rates))
    
    hpd50 = hpd(sample_day,percentile=50)
    hpd90 = hpd(sample_day,percentile=90)
    
    q5[i]  = hpd90[0]
    q95[i] = hpd90[1]
    q25[i] = hpd50[0]
    q75[i] = hpd50[1]

window = 3
order = 2

mean_s = savgol_filter(mean, window, order)
q5_s = savgol_filter(q5, window, order)
q25_s = savgol_filter(q25, window, order)
q75_s = savgol_filter(q75, window, order)
q95_s = savgol_filter(q95, window, order)

f_mean = interpolate.interp1d(jd_span, mean_s, fill_value="extrapolate", kind=3)
f_q5 = interpolate.interp1d(jd_span, q5_s, fill_value="extrapolate", kind=3)
f_q25 = interpolate.interp1d(jd_span, q25_s, fill_value="extrapolate", kind=3)
f_q75 = interpolate.interp1d(jd_span, q75_s, fill_value="extrapolate", kind=3)
f_q95 = interpolate.interp1d(jd_span, q95_s, fill_value="extrapolate", kind=3)

v_f_mean = np.vectorize(f_mean)
v_f_q5 = np.vectorize(f_q5)
v_f_q25 = np.vectorize(f_q25)
v_f_q75 = np.vectorize(f_q75)
v_f_q95 = np.vectorize(f_q95)

fine_jd_span = np.arange(2459028.5,2459572.5,1)
fine_date_span = np.zeros(len(fine_jd_span),dtype=dt.date)
for i in range(len(fine_jd_span)):
    jd = fine_jd_span[i] 
    fine_date_span[i] = jd2date(jd)
    
fine_mean = v_f_mean(fine_jd_span)
fine_q5 = v_f_q5(fine_jd_span)
fine_q25 = v_f_q25(fine_jd_span)
fine_q75 = v_f_q75(fine_jd_span)
fine_q95 = v_f_q95(fine_jd_span)


# --------- plotting detected flux

#read CNN csv into np arrays and disregard nans
dust_counts = pd.read_csv("cnn.txt",delim_whitespace=True)
flux = np.array(dust_counts["daily_dust_count"])
#maybe add stdev? maybe not
years = np.zeros(0,dtype=int)
months = np.zeros(0,dtype=int)
days = np.zeros(0,dtype=int)
time_scanned = np.zeros(0,dtype=int)

# correcting for exposure
for date in dust_counts["days"]:
    years = np.append(years,int(date[0:4]))
    months = np.append(months,int(date[5:7]))
    days = np.append(days,int(date[8:10]))
    YYYYMMDD = str(int(date[0:4]))+str(int(date[5:7])).zfill(2)+str(int(date[8:10])).zfill(2)
    try:
        online = False
        cdf_file = fetch(YYYYMMDD,'data',"tds_stat","_rpw-tds-surv-stat_",["V04.cdf","V03.cdf","V02.cdf","V01.cdf"],online)
        time_scanned = np.append(time_scanned,sum(cdf_file.varget("snapshot_len")*cdf_file.varget("SN_NR_EVENTS")/cdf_file.varget("sampling_rate")/3600))
        #time in hours = samples per window * number of windows / ( sampling rate per second * seconds / hour )
    except:
        time_scanned = np.append(time_scanned,np.nan)
        
mask = ~np.isnan(flux) * ~np.isnan(time_scanned)
flux = flux[mask]
    
jd_flux = np.zeros(len(flux))
for i in range(len(flux)):
    jd_flux[i] = date2jd(dt.date(years[mask][i],months[mask][i],days[mask][i]))

flux_date = np.zeros(len(jd_flux),dtype=dt.date)
for i in range(len(jd_flux)):
    flux_date[i] = jd2date(jd_flux[i]).date()

time_scanned_masked = time_scanned[mask]
flux_cnn = flux / (time_scanned_masked)

#%% plotting
fig,ax = plt.subplots()
ax.scatter(flux_date, flux_cnn*24, color = u"#F15025", s=1, zorder = 1000)#, label = "CNN impacts")
ax.plot(fine_date_span,fine_mean*24,color=u"#57467B",label="Mean")
#ax.fill_between(fine_date_span, fine_q25*24, fine_q75*24, linewidth = 0, color = u"#97873a", alpha = 1,label="50\%", zorder = 1)
ax.fill_between(fine_date_span, fine_q5*24, fine_q95*24, linewidth = 0, color = u"#3DDC97", alpha = 1,label="90\%", zorder = 0)

ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.set_ylim(0,520)
ax.set_xlim(np.min(fine_date_span),np.max(fine_date_span))
ax.tick_params(axis='x',which="minor",bottom=True,top=True)
ax.tick_params(axis='x',labelrotation=45 )
ax.legend(loc=(0.08,0.65),fontsize="small",frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax.set_ylabel(r"Impact rate [$day^{-1}$]")
ax.set_aspect((543)/(520*1.618))
ax2 = ax.twinx()
ax2.set_ylim(0,2.708333)
ax2.set_ylabel(r"Impact rate [$m^{-2} h^{-1}$]")
fig.tight_layout(pad=0.5)
fig.savefig('figs\\modelled_flux_rate.pdf', format='pdf', dpi=600)
fig.show()


#%% flux of beta onto a unif sphere
# or into the Earth atmosphere

sample_size_mu = 100000
add_bg = True
add_hyperbolic = False

def mu(b1, b2, c1, c2, v1, r, vr, vt):
    return ( (((vr-v1)**2+(vt-(12.5*0.75/r))**2)**0.5)/50 )**(b1)*r**(b2)*c1 + c2

rates = np.zeros(0)

for j in range(sample_size_mu):
        ib1 = b1[j]
        ib2 = b2[j]
        ic1 = c1[j]
        ic2 = c2[j]
        iv1 = v1[j]
        imu = mu(ib1,ib2,ic1*add_hyperbolic,add_bg*ic2,iv1,1,0,0)#29.5,0)
        rates = np.append(rates,imu)
        
plt.hist(rates,bins=100)
plt.show()

print("-------------")
print("Rate [/h]:")
print("mean = " + str(np.mean(rates)))
print("stdev = " + str(np.sqrt(np.var(rates))))
print("-------------")
print("Rate [*10^4 /m^2 /s]:")
print("mean = " + str(np.mean(rates)*1e4/(8*3600)))
print("stdev = " + str(np.sqrt(np.var(rates)*1e4/(8*3600))))
print("-------------")



#%% combinations of covariances
 
parameters = [b1[:,0], b2[:,0], c1[:,0], c2[:,0], v1[:,0]]
names = [r"$\epsilon_v$",r"$\epsilon_r$",r"$\lambda_\beta$",r"$\lambda_{bg}$",r"$\nu_r$"]

fig, axs = plt.subplots(5, 5)
for ix in range(5):
    line = ""
    for iy in range(5):
        axs[ix,iy].hist2d(parameters[iy],parameters[ix],cmap="YlOrBr",bins=50)
        axs[ix,iy].tick_params(axis='y',labelleft=False,labelright=False)
        axs[ix,iy].tick_params(axis='x',labeltop=False,labelbottom=False)
        line = line+str(np.round(np.corrcoef(parameters[iy],parameters[ix])[0,1],3))+" "
        if iy==ix:
            p = mpl.patches.Rectangle((-100, -100), 300, 300, fill=True, ec=None, color="white", alpha=1, zorder = 5)
            axs[ix,iy].add_artist(p)
            axs[ix,iy].spines["top"].set_visible(False)
            axs[ix,iy].spines["bottom"].set_visible(False)
            axs[ix,iy].spines["right"].set_visible(False)
            axs[ix,iy].spines["left"].set_visible(False)
            axs[ix,iy].tick_params(axis='y',labelleft=False,labelright=False)
            axs[ix,iy].tick_params(axis='x',labeltop=False,labelbottom=False)
            axs[ix,iy].tick_params(axis='x',which="minor",bottom=False,top=False)
            axs[ix,iy].tick_params(axis='y',which="minor",left=False,right=False)
            axs[ix,iy].tick_params(axis='x',which="major",bottom=False,top=False)
            axs[ix,iy].tick_params(axis='y',which="major",left=False,right=False)
        if ix==4:
            axs[ix,iy].tick_params(axis='x',labelbottom=True,labelsize="x-small")
            axs[ix,iy].set_xlabel(names[iy])
            axs[ix,iy].tick_params(axis='x',labelrotation=90)
            axs[ix,iy].xaxis.set_label_coords(0.5,-1.2)
        if iy==0:
            axs[ix,iy].tick_params(axis='y',labelleft=True,labelsize="x-small")
            axs[ix,iy].set_ylabel(names[ix])
            axs[ix,iy].yaxis.set_label_coords(-0.65,0.5)
    print(line)
fig.tight_layout(pad=0.5)
fig.savefig('figs\\all_covariances.pdf', format='pdf', dpi=600)
fig.show()

#%% covariance v1 c2

a = np.histogram2d(c2[:,0],v1[:,0],bins=200)
x_start = np.argmin(np.abs(a[1]-1))
x_end = np.argmin(np.abs(a[1]-2.25))
x = a[1][x_start:x_end]
y = np.zeros(len(x))
for i in range(len(x)):
    y[i] = np.sum(a[0][x_start+i,:]*((a[2][:-1]+a[2][1:])/2))/np.sum(a[0][x_start+i,:])

fig = plt.figure(figsize=((3.37,2.08)))
gs = fig.add_gridspec(1,12, hspace=.6)
axs = gs.subplots()
ax = plt.subplot(gs[0, :11])

#fig, ax = plt.subplots()
hst = ax.hist2d(c2[:,0],v1[:,0],cmap="YlGnBu",bins=200)
ax.plot(x,y,c="red")
ax.set_xlim(0.8,2.4)
ax.set_ylim(40,90)
#ax.set_aspect((1.6)/(50*1.618))

#fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.colorbar(hst[3],cax=axs[11])

q = ax.get_position()
#ax.set_position(pos=[0.125, 0.1, 0.8, 0.4])

ax.set_ylabel(r"Posterior of $\nu_r [km/s]$")
ax.set_xlabel(r"Posterior of $\lambda_{bg} [h^{-1}]$")
fig.tight_layout(pad=0.5)
fig.savefig('figs\\covariance_c2_v1.pdf', format='pdf', dpi=600, bbox_inches="tight")
fig.show()

print(np.corrcoef(c2[:,0],v1[:,0]))


#%% means and FWHMs of posteriors

print("b1")
print("mean = " + str(np.mean(b1)))
print("stdev = " + str(np.sqrt(np.var(b1))))
print("-------------")
print("b2")
print("mean = " + str(np.mean(b2)))
print("stdev = " + str(np.sqrt(np.var(b2))))
print("-------------")
print("c1")
print("mean = " + str(np.mean(c1)))
print("stdev = " + str(np.sqrt(np.var(c1))))
print("-------------")
print("c2")
print("mean = " + str(np.mean(c2)))
print("stdev = " + str(np.sqrt(np.var(c2))))
print("-------------")
print("v1")
print("mean = " + str(np.mean(v1)))
print("stdev = " + str(np.sqrt(np.var(v1))))
print("-------------")



