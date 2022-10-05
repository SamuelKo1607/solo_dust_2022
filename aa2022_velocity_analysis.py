import sys
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 600
import aa2022_figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
from aa2022_read_ephemeris import load_ephemeris
from aa2022_conversions import jd2date
from aa2022_conversions import date2jd
from aa2022_conversions import date2unixtime
from aa2022_conversions import unix2jd
from aa2022_conversions import jd2unix
from aa2022_conversions import unix2date
from scipy import interpolate 
from scipy import optimize
from scipy import signal
from scipy.fft import fft
import datetime as dt
import os
import csv
from aa2022_download import fetch

au = 149597870.7 #astronomical unit, km
GM = 1.327e20 #gravity of Sun, m3 s-2
b, d_b = 3.94, 0.34 #exponent of Q ~ v and distribution of masses respectively
S = 8 #scacecraft surface, m^2

#dates and gravity assists
start_date = dt.date(2020,6,29)
gravity_assists = [dt.date(2020,12,27),
                   dt.date(2021,8,9),
                   dt.date(2021,11,27)]
end_date = dt.date(2021,12,16)
terminators = [start_date, *gravity_assists, end_date]
segments = 3

trustworthy_intervals = [[0.836,0.952],[0.620,0.921],[0.595,0.713]]

#load epehemeris
jd_ephem, hae_r, hae_v, hae_phi, radial_v, tangential_v = load_ephemeris("solo")
heliocentric_distance = np.sqrt(hae_r[:,0]**2+hae_r[:,1]**2+hae_r[:,2]**2)/au #in au
f_heliocentric_distance = interpolate.interp1d(jd_ephem,heliocentric_distance,fill_value="extrapolate",kind=3)
f_rad_v = interpolate.interp1d(jd_ephem,radial_v,fill_value="extrapolate",kind=3)
f_tan_v = interpolate.interp1d(jd_ephem,tangential_v,fill_value="extrapolate",kind=3)

#read CNN csv into np arrays and disregard nans
dust_counts = pd.read_csv("cnn.txt",delim_whitespace=True)
flux = np.array(dust_counts["daily_dust_count"])
flux_std = np.array(dust_counts["daily_dust_std"])
years = np.zeros(0,dtype=int)
months = np.zeros(0,dtype=int)
days = np.zeros(0,dtype=int)
time_scanned = np.zeros(0,dtype=int)

for date in dust_counts["days"]:
    years = np.append(years,int(date[0:4]))
    months = np.append(months,int(date[5:7]))
    days = np.append(days,int(date[8:10]))
    # correcting for exposure
    YYYYMMDD = str(int(date[0:4]))+str(int(date[5:7])).zfill(2)+str(int(date[8:10])).zfill(2)
    try:
        online = False
        cdf_file = fetch(YYYYMMDD,'data',"tds_stat","_rpw-tds-surv-stat_",["V04.cdf","V03.cdf","V02.cdf","V01.cdf"],online)
        scan_mask = cdf_file.varget("snapshot_len")<=262144
        if min(scan_mask) == False:
            time_scanned = np.append(time_scanned,np.average(cdf_file.varget("snapshot_len")[scan_mask])*sum(cdf_file.varget("SN_NR_EVENTS")/cdf_file.varget("sampling_rate")/3600))
        else:
            time_scanned = np.append(time_scanned,sum(cdf_file.varget("snapshot_len")*cdf_file.varget("SN_NR_EVENTS")/cdf_file.varget("sampling_rate")/3600))
        #time in hours = samples per window * number of windows / ( sampling rate per second * seconds / hour )
    except:
        time_scanned = np.append(time_scanned,np.nan)

mask = ~np.isnan(flux) * ~np.isnan(time_scanned)
flux_masked = flux[mask]
jd_flux = np.zeros(len(flux_masked))
time_scanned_masked = time_scanned[mask]
for i in range(len(flux_masked)):
    jd_flux[i] = date2jd(dt.date(years[mask][i],months[mask][i],days[mask][i]))

#make bootstrap replications
replications = 1000
replicated_flux_masked = np.zeros((replications,len(flux_masked)))   #empty array for replications [replication,day]
for i in range(len(flux_masked)):
    #the first will be actually observed
    replicated_flux_masked[0,i] = flux_masked[i]
    #fill the rest with random
    replicated_flux_masked[1:,i] = np.random.poisson(lam = flux_masked[i], size = replications-1)

#mornalization to detection time    
flux_masked_normalized = flux_masked / time_scanned_masked   #correction for duty cycle, now in /h
replicated_flux_masked_normalized = replicated_flux_masked / time_scanned_masked

#define nonparametric fitting
def f_expected_flux_both_unparametric(  heliocentric,       #data, negative for inbound
                                        flux,               #flux, same shape
                                        sigma,              #stdev, same shape
                                        peri,               #perihelion
                                        aphe,               #aphelion
                                        fwhm,               #fitting kernel
                                        r,                  #independent variable, negative for inbound
                                        linked = False):    # whether to use information from the other leg        
            #gaussian    
            w_same = np.exp(-abs((heliocentric[heliocentric*r>0]-r)/(fwhm/2.355))**2)
            
            #lorentzian
            #w_same = (1/np.pi)*(1 / ((heliocentric[heliocentric*r>0]-r)**2 + 0.03**2))
            
            #linear
            #w_same = np.maximum(fwhm-np.abs(heliocentric[heliocentric*r>0]-r),0)
            
            #gaussian only
            w_other = linked * np.exp(-((np.minimum( abs(abs(r)+abs(heliocentric[heliocentric*r<0])-2*peri),
                                                abs(2*aphe-abs(r)-abs(heliocentric[heliocentric*r<0])))
                                    )/(fwhm/2.355))**2)
            
            return (sum(w_same*flux[heliocentric*r>0]*sigma[heliocentric*r>0])
                    +sum(w_other*flux[heliocentric*r<0]*sigma[heliocentric*r<0])
                    ) / (sum(w_same*sigma[heliocentric*r>0])+sum(w_other*sigma[heliocentric*r<0]))

#prepare arrays for inferred velocity for each replication
resolution = 249
r_heliocentric = np.zeros((segments,resolution))
flux_in = np.zeros((replications,segments,resolution))
flux_out = np.zeros((replications,segments,resolution))

#arrays for scatters 
scatter_in = []
scatter_out = []

#iterate branches
for segment in range(segments):
    start_date = terminators[segment]+dt.timedelta(days=1)
    end_date = terminators[segment+1]
    start_unixtime = date2unixtime(start_date)
    end_unixtime = date2unixtime(end_date)
    
    #get grid for this orbit
    unixtimespan = np.arange(start_unixtime, end_unixtime, 86400)
    heliocentric = f_heliocentric_distance(unix2jd(unixtimespan))
    radial = f_rad_v(unix2jd(unixtimespan))
    tangential = f_tan_v(unix2jd(unixtimespan))
    mask_inward = radial<0
    
    #get the perihelion and aphelion distance in AU
    if 0 < np.argmin(heliocentric) < len(heliocentric)-1:
        perihelion = min(heliocentric)
        v_p = tangential[np.argmin(heliocentric)]
    else:
        v_a = tangential[np.argmax(heliocentric)]
        aphelion = max(heliocentric)
        k1 = (1e3*v_a)**2 - 2*GM/(aphelion*au*1e3)
        k2 = 2*GM
        k3 = -(1e3*v_a * aphelion*au*1e3)**2
        
        x1 = (-k2+np.sqrt(k2**2-4*k1*k3))/(2*k1) /(1e3*au)
        x2 = (-k2-np.sqrt(k2**2-4*k1*k3))/(2*k1) /(1e3*au)
        
        perihelion = min(x1,x2)
        v_p = v_a * aphelion / perihelion
        
    if 0 < np.argmax(heliocentric) < len(heliocentric)-1:
        aphelion = max(heliocentric)
        v_a = tangential[np.argmax(heliocentric)]
        v_p = v_a * aphelion / perihelion
    else:
        v_p = tangential[np.argmin(heliocentric)]
        perihelion = min(heliocentric)
        k1 = (1e3*v_p)**2 - 2*GM/(perihelion*au*1e3)
        k2 = 2*GM
        k3 = -(1e3*v_p * perihelion*au*1e3)**2
        
        x1 = (-k2+np.sqrt(k2**2-4*k1*k3))/(2*k1) /(1e3*au)
        x2 = (-k2-np.sqrt(k2**2-4*k1*k3))/(2*k1) /(1e3*au)
        
        aphelion = max(x1,x2)
        v_a = v_p * perihelion / aphelion
    
    #get the grid that we will use further
    r_heliocentric[segment,:] = np.arange(perihelion,aphelion+(aphelion-perihelion)/(resolution-1),step=(aphelion-perihelion)/(resolution-1))
    
    #format the detections
    mask_this_orbit = np.arange(len(jd_flux))[(jd_flux>date2jd(start_date))*(jd_flux<date2jd(end_date))]
    jd_this_orbit = jd_flux[mask_this_orbit]
    mask_flux_in_this_orbit = f_rad_v(jd_this_orbit)<0
    heliocentric_in_this_orbit =  f_heliocentric_distance(jd_this_orbit[mask_flux_in_this_orbit==1])
    heliocentric_out_this_orbit = f_heliocentric_distance(jd_this_orbit[mask_flux_in_this_orbit==0])
    
    #fill arrays of scatters for plotting
    scatter_in.append([heliocentric_in_this_orbit,flux_masked_normalized[mask_this_orbit][mask_flux_in_this_orbit==1]])
    scatter_out.append([heliocentric_out_this_orbit,flux_masked_normalized[mask_this_orbit][mask_flux_in_this_orbit==0]])
    
    #iterate through the replications and heliocentric grid
    for i in range(replications):
        flux_replication_in = replicated_flux_masked_normalized[i,:][mask_this_orbit][mask_flux_in_this_orbit==1]
        flux_replication_out = replicated_flux_masked_normalized[i,:][mask_this_orbit][mask_flux_in_this_orbit==0]
        fwhm = np.random.uniform(0.1,0.2)
        for j in range(resolution):
            flux_in[i,segment,j] = f_expected_flux_both_unparametric(
                                                    np.append(-heliocentric_in_this_orbit,heliocentric_out_this_orbit),
                                                    np.append(flux_replication_in,flux_replication_out),
                                                    np.ones(len(np.append(flux_replication_in,flux_replication_out))),
                                                    perihelion,
                                                    aphelion,
                                                    fwhm,
                                                    -r_heliocentric[segment,j])
            flux_out[i,segment,j] = f_expected_flux_both_unparametric(
                                                    np.append(-heliocentric_in_this_orbit,heliocentric_out_this_orbit),
                                                    np.append(flux_replication_in,flux_replication_out),
                                                    np.ones(len(np.append(flux_replication_in,flux_replication_out))),
                                                    perihelion,
                                                    aphelion,
                                                    fwhm,
                                                    r_heliocentric[segment,j])  

#plotting
fig = plt.figure(figsize=(3.6,7))
gs = fig.add_gridspec(3, hspace=.05)
ax = gs.subplots()

for segment in range(segments):
    mask_in = (r_heliocentric[segment,:]>np.min(scatter_in[segment][0])) * (r_heliocentric[segment,:]<np.max(scatter_in[segment][0]))
    mask_out = (r_heliocentric[segment,:]>np.min(scatter_out[segment][0])) * (r_heliocentric[segment,:]<np.max(scatter_out[segment][0]))
    
    ax[segment].scatter(scatter_in[segment][0],scatter_in[segment][1], color="black",label="Inbound")
    ax[segment].scatter(scatter_out[segment][0],scatter_out[segment][1],color="red",label="Outbound")
    
    #the original sample
    #ax[segment].plot(r_heliocentric[segment,:][mask_in],flux_in[0,segment,:][mask_in],color="green",zorder=50)
    #ax[segment].plot(r_heliocentric[segment,:][mask_out],flux_out[0,segment,:][mask_out],color="green",zorder=50)
    
    ax[segment].fill_between(r_heliocentric[segment,:][mask_in],np.quantile(flux_in[:,segment,:],.05,axis=0)[mask_in],np.quantile(flux_in[:,segment,:],.95,axis=0)[mask_in],color="black",alpha=0.2)
    ax[segment].fill_between(r_heliocentric[segment,:][mask_in],np.quantile(flux_in[:,segment,:],.25,axis=0)[mask_in],np.quantile(flux_in[:,segment,:],.75,axis=0)[mask_in],color="black",alpha=0.5)
    ax[segment].fill_between(r_heliocentric[segment,:][mask_out],np.quantile(flux_out[:,segment,:],.05,axis=0)[mask_out],np.quantile(flux_out[:,segment,:],.95,axis=0)[mask_out],color="red",alpha=0.2)
    ax[segment].fill_between(r_heliocentric[segment,:][mask_out],np.quantile(flux_out[:,segment,:],.25,axis=0)[mask_out],np.quantile(flux_out[:,segment,:],.75,axis=0)[mask_out],color="red",alpha=0.5)
    
    ax[segment].plot(r_heliocentric[segment,:][mask_in],np.average(flux_in[:,segment,:],axis=0)[mask_in],color="black",ls="solid")
    ax[segment].plot(r_heliocentric[segment,:][mask_out],np.average(flux_out[:,segment,:],axis=0)[mask_out],color="red",ls="solid")
    
    #ax[segment].plot(r_heliocentric[segment,:][mask_in*mask_out],np.average(flux_in[:,segment,:],axis=0)[mask_in*mask_out],color="black",ls="solid")
    #ax[segment].plot(r_heliocentric[segment,:][mask_in*mask_out],np.average(flux_out[:,segment,:],axis=0)[mask_in*mask_out],color="red",ls="solid")
    
    ax[segment].fill_between(trustworthy_intervals[segment],[-5,-5],[100,100],alpha=0.1,zorder=-5,color=u"#283845",lw=0)
    
    ax[segment].set_ylim(-1,22)
    ax[segment].set_xlim(0.48,1.02)
    ax[segment].set_aspect(0.54/(23*1.618))                 
    ax[segment].xaxis.set_ticks([0.5,0.6,0.7,0.8,0.9,1.0])
    ax[segment].text(0.51,19,"Branch "+str(segment+1),fontsize="large",backgroundcolor="white",zorder=10)

ax[2].set_xlabel("Heliocentric distance [AU]")
ax[1].set_ylabel("Detection rate [/hour]")
ax[0].legend(loc=1,fontsize="small",edgecolor="white")
ax[0].tick_params(axis='x',labeltop=False,labelbottom=False)
ax[1].tick_params(axis='x',labeltop=False,labelbottom=False)
fig.savefig("figs\\nonparam_fit_all_boostrap.pdf", format='pdf', dpi=600, bbox_inches="tight")
fig.show()  

#%% velocity estimate - plot by backgrounds

bdb = 1.3                           #1.0, 1.3, 1.6
correct_beta_tangential = 12        #6, 12, 22
cols = [u'#C65B7C', u'#ff7f0e', u'#379634', u'#d62728', u'#445E93']
min_radial_velocity = 5


fig,ax = plt.subplots()

for background in [0,2,4]:
    
    #prepare empty arrays
    heliocentric_distance = []
    inferred_velocity = []
    r_spans = []
    trustworthies = []
    
    for segment in range(segments):
        
        #distance grid
        mask_in = (r_heliocentric[segment,:]>np.min(scatter_in[segment][0])) * (r_heliocentric[segment,:]<np.max(scatter_in[segment][0]))
        mask_out = (r_heliocentric[segment,:]>np.min(scatter_out[segment][0])) * (r_heliocentric[segment,:]<np.max(scatter_out[segment][0]))
        r_span = r_heliocentric[segment,:][mask_in*mask_out]
        
        #decide whether to use each r_span point based on if inward and aoutward are available in its vicinity
        trustworthy = np.zeros(len(r_span),dtype=bool)
        for i in range(len(r_span)):
            distance_to_inward_information = np.min(np.abs(scatter_in[segment][0]-r_span[i]))
            distance_to_outward_information = np.min(np.abs(scatter_out[segment][0]-r_span[i]))
            if max([distance_to_inward_information,distance_to_outward_information]) < 0.02 and r_span[i] > 0.58:
                trustworthy[i] = True       
        r_spans.append(r_span)
        trustworthies.append(trustworthy)
        
        #get velocities as a function of r
        start_date = terminators[segment]+dt.timedelta(days=1)
        end_date = terminators[segment+1]
        start_unixtime = date2unixtime(start_date)
        end_unixtime = date2unixtime(end_date)
        unixtimespan = np.arange(start_unixtime, end_unixtime, 86400)
        heliocentric = f_heliocentric_distance(unix2jd(unixtimespan))
        radial = f_rad_v(unix2jd(unixtimespan))
        tangential = f_tan_v(unix2jd(unixtimespan))
        
        rad_this_orbit_coef = np.polyfit(heliocentric, np.abs(radial),12)
        tan_this_orbit_coef = np.polyfit(heliocentric, tangential,12)
        f_rad_this_orbit = np.poly1d(rad_this_orbit_coef)
        f_tan_this_orbit = np.poly1d(tan_this_orbit_coef)
        #def t_totv_this_orbit(heliocentric):
        #    return np.sqrt( f_rad_this_orbit(heliocentric)**2 + f_tan_this_orbit(heliocentric)**2 )
        
        rad_span = f_rad_this_orbit(r_span)
        tan_span = f_tan_this_orbit(r_span) - (correct_beta_tangential*0.75)/r_span
        tot_span = (rad_span**2 + tan_span**2)**0.5
        
        #prepare arrays for output
        heliocentric_distance.append(r_span[rad_span>min_radial_velocity])
        velocities = np.zeros((replications,len(r_span[rad_span>min_radial_velocity])))
        
        for i in range(replications):
            
            in_span = np.maximum(flux_in[i,segment,:][mask_in*mask_out] - background,0)
            out_span = np.maximum(flux_out[i,segment,:][mask_in*mask_out] - background,0)
            
            #velocity estimate
            D_hat = rad_span**2 * (in_span**(2/(1+bdb)) + out_span**(2/(1+bdb)))**2 - tot_span**2 * (in_span**(2/(1+bdb)) - out_span**(2/(1+bdb)))**2
            v_estimate = (rad_span * (in_span**(2/(1+bdb)) + out_span**(2/(1+bdb))) + np.sqrt(D_hat) )/((in_span**(2/(1+bdb)) - out_span**(2/(1+bdb))))
            
            #fill array
            velocities[i,:] = v_estimate[rad_span>min_radial_velocity]
            
            #individual traces
            #plt.plot(r_span[rad_span>min_radial_velocity],v_estimate[rad_span>min_radial_velocity],color=cols[segment],alpha=0.02)
    
        inferred_velocity.append(velocities)
           
    #plotting aggreagates
    ax.plot([-1,0],[-1,0],lw=1,c=cols[background],label = r"$\lambda_{bg} =$ "+str(background),zorder=100)
    for segment in range(segments):
        ax.plot(heliocentric_distance[segment],np.nanquantile(inferred_velocity[segment][:,:],0.5,axis=0),color=cols[background],zorder=10*(segment+1)-100)
        #ax.plot(heliocentric_distance[segment],np.nanmean(inferred_velocity[segment][:,:],axis=0),color=cols[segment],zorder=10*(segment+1)-100)
        ax.fill_between(heliocentric_distance[segment],np.nanquantile(inferred_velocity[segment][:,:],0.25,axis=0),np.nanquantile(inferred_velocity[segment][:,:],0.75,axis=0),color=cols[background],lw=0,alpha=0.3,zorder=10*(segment+1)+1-100)
        ax.fill_between(r_spans[segment], 1000*(1-trustworthies[segment]), color="white", lw=1, alpha=1,zorder=10*(segment+1)+2-100)


    
ax.set_ylim(0,170)
ax.set_xlim(0.48,1.02)
ax.tick_params(
        axis='y',
        which='major',    
        labelright=True) 
ax.set_xlabel("Heliocentric distance [AU]")
ax.set_ylabel("Dust radial velocity [km/s]")
ax.legend(frameon=False,loc=3,fontsize="small",ncol=3)

ax.text(0.95,70,"Branch 1", fontsize="small", color=u'#C65B7C', rotation=-10, ha="center", va="bottom",bbox=dict(boxstyle="square",fc="none",ec="none"))
ax.text(0.778,110,"Branch 2", fontsize="small", color=u'#C65B7C', rotation=-35,ha="center", va="top",bbox=dict(boxstyle="square",fc="none",ec="none"))
ax.text(0.68,130,"Branch 3", fontsize="small", color=u'#C65B7C', rotation=-15,ha="center", va="bottom",bbox=dict(boxstyle="square",fc="none",ec="none"))


fig.savefig('figs\\nonparametric_profile_background_bootstrap.pdf', format='pdf', dpi=600, bbox_inches="tight")
fig.show()

#%% velocity estimate - plot by backgrounds, 3 at a time

bdb = 1.6                                   #1.0, 1.3, 1.6
correct_beta_tangentials = [6,12,22]        #6, 12, 22
r0s = [0.02,0.1,0.3]
cols = [u'#C65B7C', u'#ff7f0e', u'#379634', u'#d62728', u'#445E93']
min_radial_velocity = 5

fig = plt.figure(figsize=((3.37,5.45)))
gs = fig.add_gridspec(3, hspace=.0)
axs = gs.subplots()

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

for j in range(len(correct_beta_tangentials)):
    correct_beta_tangential = correct_beta_tangentials[j]
    ax = axs[j]
    
    for background in [0,2,4]:
        
        #prepare empty arrays
        heliocentric_distance = []
        inferred_velocity = []
        r_spans = []
        trustworthies = []
        
        for segment in range(segments):
            
            #distance grid
            mask_in = (r_heliocentric[segment,:]>np.min(scatter_in[segment][0])) * (r_heliocentric[segment,:]<np.max(scatter_in[segment][0]))
            mask_out = (r_heliocentric[segment,:]>np.min(scatter_out[segment][0])) * (r_heliocentric[segment,:]<np.max(scatter_out[segment][0]))
            r_span = r_heliocentric[segment,:][mask_in*mask_out]
            
            #decide whether to use each r_span point based on if inward and aoutward are available in its vicinity
            trustworthy = np.zeros(len(r_span),dtype=bool)
            for i in range(len(r_span)):
                distance_to_inward_information = np.min(np.abs(scatter_in[segment][0]-r_span[i]))
                distance_to_outward_information = np.min(np.abs(scatter_out[segment][0]-r_span[i]))
                if max([distance_to_inward_information,distance_to_outward_information]) < 0.02 and r_span[i] > 0.58:
                    trustworthy[i] = True       
            r_spans.append(r_span)
            trustworthies.append(trustworthy)
            
            #get velocities as a function of r
            start_date = terminators[segment]+dt.timedelta(days=1)
            end_date = terminators[segment+1]
            start_unixtime = date2unixtime(start_date)
            end_unixtime = date2unixtime(end_date)
            unixtimespan = np.arange(start_unixtime, end_unixtime, 86400)
            heliocentric = f_heliocentric_distance(unix2jd(unixtimespan))
            radial = f_rad_v(unix2jd(unixtimespan))
            tangential = f_tan_v(unix2jd(unixtimespan))
            
            rad_this_orbit_coef = np.polyfit(heliocentric, np.abs(radial),12)
            tan_this_orbit_coef = np.polyfit(heliocentric, tangential,12)
            f_rad_this_orbit = np.poly1d(rad_this_orbit_coef)
            f_tan_this_orbit = np.poly1d(tan_this_orbit_coef)

            
            rad_span = f_rad_this_orbit(r_span)
            tan_span = f_tan_this_orbit(r_span) - (correct_beta_tangential*0.75)/r_span
            tot_span = (rad_span**2 + tan_span**2)**0.5
            
            #prepare arrays for output
            heliocentric_distance.append(r_span[rad_span>min_radial_velocity])
            velocities = np.zeros((replications,len(r_span[rad_span>min_radial_velocity])))
            
            for i in range(replications):
                
                in_span = np.maximum(flux_in[i,segment,:][mask_in*mask_out] - background,0)
                out_span = np.maximum(flux_out[i,segment,:][mask_in*mask_out] - background,0)
                
                #velocity estimate
                D_hat = rad_span**2 * (in_span**(2/(1+bdb)) + out_span**(2/(1+bdb)))**2 - tot_span**2 * (in_span**(2/(1+bdb)) - out_span**(2/(1+bdb)))**2
                v_estimate = (rad_span * (in_span**(2/(1+bdb)) + out_span**(2/(1+bdb))) + np.sqrt(D_hat) )/((in_span**(2/(1+bdb)) - out_span**(2/(1+bdb))))
                
                #fill array
                velocities[i,:] = v_estimate[rad_span>min_radial_velocity]
                
                #individual traces
                #plt.plot(r_span[rad_span>min_radial_velocity],v_estimate[rad_span>min_radial_velocity],color=cols[segment],alpha=0.02)
        
            inferred_velocity.append(velocities)
               
        #plotting aggreagates
        ax.plot([-1,0],[-1,0],lw=1,c=cols[background],label = r"$\lambda_{bg} =$ "+str(background),zorder=100)
        for segment in range(segments):
            ax.plot(heliocentric_distance[segment],np.nanquantile(inferred_velocity[segment][:,:],0.5,axis=0),color=cols[background],zorder=10*(segment+1)-100)
            #ax.plot(heliocentric_distance[segment],np.nanmean(inferred_velocity[segment][:,:],axis=0),color=cols[segment],zorder=10*(segment+1)-100)
            ax.fill_between(heliocentric_distance[segment],np.nanquantile(inferred_velocity[segment][:,:],0.25,axis=0),np.nanquantile(inferred_velocity[segment][:,:],0.75,axis=0),color=cols[background],lw=0,alpha=0.3,zorder=10*(segment+1)+1-100)
            ax.fill_between(r_spans[segment], 1000*(1-trustworthies[segment]), color="white", lw=1, alpha=1,zorder=10*(segment+1)+2-100)
    
    ax.set_ylim(0,170)
    ax.set_xlim(0.48,1.02)
    ax.tick_params(
            axis='y',
            which='major',    
            labelright=True) 
    ax.legend(frameon=False,loc=3,fontsize="x-small",ncol=3)
    
    ax.text(0.8,150,r"$\alpha \delta =$ "+str(bdb),ha="center",va="center",bbox=dict(boxstyle="square",fc="white",ec="white"))
    ax.text(0.8,130,r"$r_0=$ "+str(r0s[j]),ha="center",va="center",bbox=dict(boxstyle="square",fc="white",ec="white"))
    
ax3.set_xlabel("Heliocentric distance [AU]")
ax2.set_ylabel("Dust radial velocity [km/s]")

fig.savefig('figs\\nonparametric_profile_background_bootstrap_alphadelta'+str(bdb)+'.pdf', format='pdf', dpi=600, bbox_inches="tight")
fig.show()

#%% velocity estimate - plot influence of other parameters

def estimate(background,bdb,correct_beta_tangential):
    est = []
    
    #points to evaluate
    points = [[0.85],[0.65,0.75,0.85],[0.65]]
    
    for segment in range(segments):
        
        #get velocities as a function of r
        start_date = terminators[segment]+dt.timedelta(days=1)
        end_date = terminators[segment+1]
        start_unixtime = date2unixtime(start_date)
        end_unixtime = date2unixtime(end_date)
        unixtimespan = np.arange(start_unixtime, end_unixtime, 86400)
        heliocentric = f_heliocentric_distance(unix2jd(unixtimespan))
        radial = f_rad_v(unix2jd(unixtimespan))
        tangential = f_tan_v(unix2jd(unixtimespan))
        rad_this_orbit_coef = np.polyfit(heliocentric, np.abs(radial),12)
        tan_this_orbit_coef = np.polyfit(heliocentric, tangential,12)
        f_rad_this_orbit = np.poly1d(rad_this_orbit_coef)
        f_tan_this_orbit = np.poly1d(tan_this_orbit_coef)
        
        for p in points[segment]:
            
            rad_span = f_rad_this_orbit(p)
            tan_span = f_tan_this_orbit(p) - (correct_beta_tangential*0.75)/p
            tot_span = (rad_span**2 + tan_span**2)**0.5
            
            r_index = np.argmin(abs(r_heliocentric[segment,:]-p))
            velocities = np.zeros(replications)
        
            for i in range(replications):
                
                in_span = np.maximum(flux_in[i,segment,r_index] - background,0)
                out_span = np.maximum(flux_out[i,segment,r_index] - background,0)
                
                #velocity estimate
                D_hat = rad_span**2 * (in_span**(2/(1+bdb)) + out_span**(2/(1+bdb)))**2 - tot_span**2 * (in_span**(2/(1+bdb)) - out_span**(2/(1+bdb)))**2
                v_estimate = (rad_span * (in_span**(2/(1+bdb)) + out_span**(2/(1+bdb))) + np.sqrt(D_hat) )/((in_span**(2/(1+bdb)) - out_span**(2/(1+bdb))))
                
                #fill array
                velocities[i] = v_estimate


            est.append(np.nanquantile(velocities,0.5))
    
    return est




x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)*100

fig = plt.figure()
gs = fig.add_gridspec(1, 3, wspace = 0)
(ax1, ax2, ax3) = gs.subplots(sharex='col', sharey='row')

ax1.set_aspect(9/130/1.618)
ax2.set_aspect(9/130/1.618)
ax3.set_aspect(9/130/1.618)

gs.update(wspace=-0.0001)
fig.tight_layout(pad=0.5)

fig.subplots_adjust(top=0.955)

colors=[u"#db5943",u"#67b045",u"#ffc60a"]

ax1.set_ylim(0,130)

ax1.title.set_text(r'$\lambda_{bg}=0$')
ax2.title.set_text(r'$\lambda_{bg}=2$')
ax3.title.set_text(r'$\lambda_{bg}=4$')

ax1.set_ylabel("Dust radial velocity [km/s]")
ax2.set_xlabel("Initial heliocentric distance [AU]")

for bg in [0,2,4]:
    if bg == 0:
        iax = ax1
    elif bg == 2:
        iax = ax2
    elif bg == 4:
        iax = ax3
    else:
        pass
    for vt in [6,12,22]:      #corresponds to [0.02,0.1,0.3] in AU
        if vt == 6:
            ix = 1
        elif vt == 12:
            ix = 2
        elif vt == 22:
            ix = 3
        else:
            pass
        for alphadelta in [1,1.3,1.6]:
            if alphadelta == 1:
                icolor = colors[0]
            elif alphadelta == 1.3:
                icolor = colors[1]
            elif alphadelta == 1.6:
                icolor = colors[2]
            else:
                pass
            
            est = estimate(bg,bdb=alphadelta,correct_beta_tangential=vt)
            print(est)
            if len(est)>4:
                fclr = icolor
            else:
                fclr = "none"
            iax.scatter(ix,np.mean(est),s=8,facecolors=fclr,edgecolors=icolor)
            
#define lines to get label correct
m1 = ax3.scatter([-1],[50],c=colors[0],label=r"$\alpha \delta = 1.0$")
m2 = ax3.scatter([-1],[50],c=colors[1],label=r"$\alpha \delta = 1.3$")
m3 = ax3.scatter([-1],[50],c=colors[2],label=r"$\alpha \delta = 1.6$")

l1 = plt.legend(handles = [m1, m2, m3],loc=1,fontsize="small",frameon=True,ncol=1,facecolor ="white",edgecolor="white")

for ax in fig.get_axes():
    ax.set_xlim(0.5,3.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_xticklabels(["dummy","0.02","0.1","0.3"])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.vlines([1.5,2.5],0,200,ls="dashed",color="gray")
    ax.title.set_size("small")
    ax.tick_params(
        axis='x',          
        which='major',   
        bottom=False,    
        top=False,      
        labelbottom=True) 
    ax.tick_params(
        axis='x',        
        which='minor',   
        bottom=False,  
        top=False,
        labelbottom=False)
    ax.tick_params(
        axis='y',
        which='minor',  
        left=False,  
        right=False) 
    ax.tick_params(
        axis='y',
        which='major',    
        length=2) 

ax3.tick_params(
        axis='y',
        which='major',    
        labelright=True) 


fig.savefig('figs\\velocity_other_parameters.pdf', format='pdf', dpi=600)
fig.show()

#%% plot velocity profiles as a function of beta and initial orbit

def orbital_velocity(r,e=0): 
    #r in AU 
    #e is eccentricity, positive means perihelion, negative means aphelion
    #result in km/s
    a = r/(1-e)
    v = np.sqrt(GM*(2/(r*au*1000) - 1/(a*au*1000)))/1000
    return v #np.sqrt(GM/(r*au*1000))/1000

def beta_velocity(a,r,e=0,beta=0,total=True,radial=True):
    # a is initial orbit in AU
    # r is position in AU
    # e is initial eccentricity
    # beta is beta-factor
    # radial motion is assumed
    # returns [radial, tangential] in km/s
    
    v_initial = orbital_velocity(a,e=e)
    
    v_total = np.sqrt( (v_initial*1000)**2 
                   + 2*GM*(1-beta)*(1/(r*au*1000)-1/(a*au*1000)) )/1000
    
    v_tangential = v_initial * a / r
    
    v_radial = np.sqrt(v_total**2 - v_tangential**2)
    
    if total:
        return v_total
    elif radial:
        return v_radial
    else:
        return v_tangential

v_beta_velocity = np.vectorize(beta_velocity)


betas = [0.5,0.6,0.7]
eccentricity = 0.0
ils = ["dotted","solid","dashed"]
origin = [0.05,0.1,0.2]
cols = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd']
span = np.arange(0.1,1.1,0.05)
plot_radial = True

fig,ax = plt.subplots()
for i in range(len(origin)):
    o = origin[i]
    for j in range(len(betas)):
        b = betas[j]
        if ils[j] == "solid":
            ax.plot(span, v_beta_velocity(o,span,beta=b,e=eccentricity,
                                           total=False,radial=plot_radial
                                           ),
                     label=r"$r_0$ = "+str(o)+r"AU",
                     ls=ils[j],
                     c=cols[i])
        else:
            ax.plot(span, v_beta_velocity(o,span,beta=b,e=eccentricity,
                                           total=False,radial=plot_radial
                                           ),
                     ls=ils[j],
                     c=cols[i])
#legend_properties = {'weight':'light','size':'x-small'}
l1 = plt.legend(fontsize="x-small",frameon=False,ncol=1,loc=2)
ls_list = []
for j in range(len(betas)):
    iplot, = ax.plot([-1,-2],[-1,-2],ls=ils[j],color="black",label=r"$\beta$ = "+str(betas[j]))
    ls_list.append(iplot)
if plot_radial:
    l2 = plt.legend(handles = ls_list, fontsize="x-small",frameon=False,ncol=1,loc=4)
ax.add_artist(l1)
#ax.add_artist(l2)
ax.set_ylim(0,100)
ax.set_xlim(0.5,1)
ax.set_xlabel("Heliocentric distance [AU]")
ax.set_aspect(0.002577*1.2)
fig.tight_layout(pad=0.5)
if plot_radial:
    ax.set_ylabel(r"$\beta$ dust radial velocity [km/s]")
    fig.savefig('figs\\beta_radial_velocity.pdf', format='pdf', dpi=600)
else:
    ax.set_ylabel(r"$\beta$ dust azimuthal velocity [km/s]")
    fig.savefig('figs\\beta_azimuthal_velocity.pdf', format='pdf', dpi=600)
fig.show()

#%% plotting spatial density exponent with overplot of eccentricity

span = np.arange(0.5,1.005,0.005)
initial_orbit = 0.1
eccentricity = 0.0
betas = [0.5,0.6,0.7,1]
#betas = [0.45,0.5,0.6,1]
plot_eccentricity_diagram = False
colors = [u"#fdca40",u"#df2935",u"#3772ff",u"#080708"]
#colors = ["teal",u"#fdca40",u"#df2935",u"#080708"]

# yellow
dust_radial_velocity_05 = v_beta_velocity(initial_orbit,span,e=eccentricity,beta=betas[0],total=False,radial=True)
synthetic_density_05 = 100*(1/(span**2))*(1/dust_radial_velocity_05)
synthetic_density_05 /= max(synthetic_density_05)
# red
dust_radial_velocity_06 = v_beta_velocity(initial_orbit,span,e=eccentricity,beta=betas[1],total=False,radial=True)
synthetic_density_06 = 100*(1/(span**2))*(1/dust_radial_velocity_06)
synthetic_density_06 /= max(synthetic_density_06)
# blue
dust_radial_velocity_07 = v_beta_velocity(initial_orbit,span,e=eccentricity,beta=betas[2],total=False,radial=True)
synthetic_density_07 = 100*(1/(span**2))*(1/dust_radial_velocity_07)
synthetic_density_07 /= max(synthetic_density_07)
# black
dust_radial_velocity_08 = v_beta_velocity(initial_orbit,span,e=eccentricity,beta=betas[3],total=False,radial=True)
synthetic_density_08 = 100*(1/(span**2))*(1/dust_radial_velocity_08)
synthetic_density_08 /= max(synthetic_density_08)

def density_function(x,c,e):
    return c*x**(e)

result1 = optimize.curve_fit(density_function, 
                            span, 
                            synthetic_density_05, 
                            p0=[0.1, -1.8])
                            #bounds = ([0,-2.000001],[100,-2]))

result2 = optimize.curve_fit(density_function, 
                            span, 
                            synthetic_density_06, 
                            p0=[0.1, -1.8])
                            #bounds = ([0,-1.6001],[100,-1.6]))

result3 = optimize.curve_fit(density_function, 
                            span, 
                            synthetic_density_07, 
                            p0=[0.1, -1.8])
                            #bounds = ([0,-1.8501],[100,-1.88]))

result4 = optimize.curve_fit(density_function, 
                            span, 
                            synthetic_density_08, 
                            p0=[0.1, -2],
                            bounds = ([0,-2.000001],[100,-2]))

fig,ax = plt.subplots()
ax.set_xlabel("Heliocentric distance [AU]")
ax.set_ylabel("Dust density [arb.u.]")
a1, = ax.plot(span,synthetic_density_05,label=r"$ \beta = $ "+str(betas[0]),color=colors[0])
a2, = ax.plot(span,synthetic_density_06,label=r"$ \beta = $ "+str(betas[1]),color=colors[1])
a3, = ax.plot(span,synthetic_density_07,label=r"$ \beta = $ "+str(betas[2]),color=colors[2])
a4, = ax.plot(span,synthetic_density_08,label=r"$ \beta = $ "+str(betas[3]),color=colors[3])
d, = ax.plot(span,density_function(span,result1[0][0],result1[0][1]),label=r"$\sim r^{"+str(np.round(result1[0][1],decimals=2))+"}$",color=colors[0],lw=2.5,ls="dashed",zorder=0,alpha=0.4)
b, = ax.plot(span,density_function(span,result2[0][0],result2[0][1]),label=r"$\sim r^{"+str(np.round(result2[0][1],decimals=2))+"}$",color=colors[1],lw=2.5,ls="dashed",zorder=0,alpha=0.4)
c, = ax.plot(span,density_function(span,result3[0][0],result3[0][1]),label=r"$\sim r^{"+str(np.round(result3[0][1],decimals=2))+"}$",color=colors[2],lw=2.5,ls="dashed",zorder=0,alpha=0.4)
f, = ax.plot(span,density_function(span,result4[0][0],result4[0][1]),label=r"$\sim r^{"+str(np.round(result4[0][1],decimals=2))+"}$",color=colors[3],lw=2.5,ls="dashed",zorder=0,alpha=0.4)
lines = ax.get_lines()
ax.legend(loc=1,frameon=True,ncol=2,facecolor ="white",edgecolor="white", fontsize="small")
ax.set_xlim(0.5,1)
ax.set_ylim(0.2,1)
ax.set_yticks([2e-1,4e-1,6e-1,8e-1,1])
ax.set_yticklabels(['0.2',"0.4","0.6","0.8","1"])
ax.set_aspect(0.5/(0.8*1.618))
ellipse = mpl.patches.Ellipse(xy=(.58, .4), 
                              width=0.25*np.sqrt(1-eccentricity**2)*0.5/0.8/1.618,
                              height=0.25, 
                              edgecolor='green', fc='None', lw=1)
if plot_eccentricity_diagram:
    ax.add_patch(ellipse)
    ax.scatter([.58],[0.4-eccentricity*0.125],c="green")
    ax.arrow(.58,(0.4-0.125),0.02,0,width=0.005,edgecolor="none",facecolor="green")
fig.tight_layout(pad=0.5)
fig.savefig('figs\\distance_exponents.pdf', format='pdf', dpi=600)
fig.show()

print(result1[0][1])

#%% plot SolO radial velocity as a function of heliocentric distance

cols=[ u'#66101F', u'#E0BE36', u'#56E39F', u'#465775']

start_date = dt.date(2020,6,29)
gravity_assists = [dt.date(2020,12,27),
                   dt.date(2021,8,9),
                   dt.date(2021,11,27)]
end_date = dt.date(2021,12,16)
terminators = [start_date, *gravity_assists, end_date]

fig, ax = plt.subplots()
for segment in range(len(terminators[:])-1):  
        start_date = terminators[segment]+dt.timedelta(days=1)
        end_date = terminators[segment+1]
        start_unixtime = date2unixtime(start_date)
        end_unixtime = date2unixtime(end_date)
        
        #get arrays of values for this orbit
        unixtimespan = np.arange(start_unixtime, end_unixtime, 86400)
        margin = 1
        radial = f_rad_v(unix2jd(unixtimespan[margin:-margin]))
        tangential = f_tan_v(unix2jd(unixtimespan[margin:-margin]))
        heliocentric = f_heliocentric_distance(unix2jd(unixtimespan[margin:-margin]))
        
        #decide if we have both directions at given AU
        both_available = np.zeros(len(heliocentric),dtype=bool)
        for i in range(len(both_available)):
            is_there = np.abs((heliocentric[i]-heliocentric)) < 0.01
            both_available[i] = np.sign(max(radial[is_there])) != np.sign(min(radial[is_there]))
        
        #plot
        ax.plot(heliocentric,abs(radial),label="Branch "+str(segment+1), alpha = 1, lw=1.7, ls="--", c=cols[segment], zorder = 1)
        ax.plot(heliocentric[both_available],abs(radial[both_available]), alpha = 1, lw=1.7, c=cols[segment], zorder = 1)
#shade out low velocities
p = mpl.patches.Rectangle((0, 0), 2, 5, fill=True, ec=None, color="white", alpha=0.7, zorder = 1)
#ax.add_artist(p)
ax.axhline(5,ls=(0,(3,4)),c="k",lw=0.5)
ax.set_ylim(0,13)
ax.set_xlim(0.45,1.05)
ax.set_xlabel("Heliocentric distance [AU]")
ax.set_ylabel("SolO heliocentric \n radial velocity [km/s]")
#fig.suptitle("SolO Radial Velocities")
ax.set_aspect((0.6)/(13*1.618))
ax.legend(fontsize="x-small",loc=8,frameon=False)
fig.tight_layout(pad=0.5)
fig.savefig('figs\\solo_state_space.pdf', format='pdf', dpi=600)
fig.show()