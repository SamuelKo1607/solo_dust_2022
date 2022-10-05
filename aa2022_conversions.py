import numpy as np
import time as t
import datetime as dt
from spacepy import pycdf

def zrot(th): #in degrees
    thrad = np.radians(th)
    c, s = np.cos(thrad), np.sin(thrad)
    return np.array(((c, -s, 0), (s, c, 0), (0,0,1)))

def gse2hae(gse,julian): #expecting 3-item vector or list in km, julian date
    gse=np.array(gse)
    if len(gse)!=3:
        raise ValueError("GSE is a 3D coordinate system, vector has to be 3 items long!")
    mjd = julian-2400000.5 #modified julian date
    #print(round(mjd))
    r0  = 1.495985e8 #in km
    T0  = (round(mjd)-51544.5)/(36525.0)
    UT  = (julian%1)*24 #UTC in hours
    M   = 357.528 + 35999.05*T0 + 0.04107*UT
    L   = 280.46 + 36000.772*T0 + 0.04107*UT
    l   = L + (1.915 - 0.0048*T0)*np.sin(np.radians(M)) + 0.02*np.sin(np.radians(2*M))
    e   = 0.016709 - 0.0000418*T0
    w   = 282.94 + 1.72*T0
    nu  = l - w
    R   = (r0*(1-e**2))/(1+e*np.cos(np.radians(nu)))
    hee = np.array([R,0,0]) + np.inner(zrot(180),gse)
    hae = np.inner(zrot(+l+180),hee)
    return hae

def gse2gae(gse,julian): #expecting 3-item vector or list in km, julian date
    gse=np.array(gse)
    if len(gse)!=3:
        raise ValueError("GSE is a 3D coordinate system, vector has to be 3 items long!")
    mjd = julian-2400000.5 #modified julian date
    #print(round(mjd))
    #r0  = 1.495985e8 #in km
    T0  = (round(mjd)-51544.5)/(36525.0)
    UT  = (julian%1)*24 #UTC in hours
    M   = 357.528 + 35999.05*T0 + 0.04107*UT
    L   = 280.46 + 36000.772*T0 + 0.04107*UT
    l   = L + (1.915 - 0.0048*T0)*np.sin(np.radians(M)) + 0.02*np.sin(np.radians(2*M))
    #e   = 0.016709 - 0.0000418*T0
    #w   = 282.94 + 1.72*T0
    #nu  = l - w
    #R   = (r0*(1-e**2))/(1+e*np.cos(np.radians(nu)))
    gae = np.inner(zrot(180),gse)
    gae = np.inner(zrot(+l+180),gae)
    return [gae[0],gae[1],gae[2]]

def tt2000_to_date(epoch): #some adapted C routine
    x=pycdf.Library()
    epoch = np.array(epoch)
    epoch = epoch.astype(np.longlong)
    x=x.v_tt2000_to_datetime(epoch)
    return x

def date_to_tt2000(date): #some adapted C routine
    x=pycdf.Library()
    x=x.v_datetime_to_tt2000(date)
    return x

def date2unixtime(date): 
    return t.mktime(date.timetuple()) 

def unix2date(unixsec):
    return dt.datetime.utcfromtimestamp(unixsec)

def unix2jd(unixsec):
    return ( unixsec / 86400.0 ) + 2440587.5

def jd2unix(jd):
    return (jd - 2440587.5) * 86400.0

def tt2000_to_unix(epoch):
    return epoch/1e9 + 946724335

def unix_to_tt2000(unixsec):
    return (unixsec - 946724335)*1e9

def jd2date(jd):
    return dt.datetime.utcfromtimestamp((jd - 2440587.5) * 86400.0)

def date2jd(date):
    return (date2unixtime(date)/ 86400.0 ) + 2440587.5

def jd_to_tt2000(jd):
    return unix_to_tt2000(jd2unix(jd))

def tt2000_to_jd(epoch):
    return unix2jd(tt2000_to_unix(epoch))

def date2YYYYMMDD(date):
    return str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)