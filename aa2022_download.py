import requests
import os
import datetime
import cdflib

#cdf_file.cdf_info()

def download(YYYYMMDD,directory,product_family,product,version_priority): #will download rpw tds from lesia
    #product example: _rpw-tds-surv-tswf-e_
    #product example: _rpw-tds-surv-stat_
    
    #product family example: tds_stat
    #product family example: tds_wf_e
    
    #version priority exapmle: ["V02.cdf","V01.cdf"]
    
    print("***** "+YYYYMMDD + ": Download initiated *****")
    a = datetime.datetime.now()
    date=str(YYYYMMDD) 
    myurl = "https://rpw.lesia.obspm.fr/roc/data/pub/solo/rpw/data/L2/"+product_family+"/"
    year = date[:4]
    month = date[4:6]
    
    success = False
    
    for version in version_priority:
        if not success:
            suffix = "_"+version
            short_file = "solo_L2"+product+date+suffix
            myfile = myurl+year+"/"+month+"/"+short_file
            r = requests.get(myfile, allow_redirects=True)
            if not str(r)=="<Response [404]>":
                success = True
                open(short_file, 'wb').write(r.content)
                print("***** "+YYYYMMDD + short_file+" downlaoded")
                print("***** "+str(round(os.path.getsize(short_file)/(1024**2),ndigits=2))+" MiB dowloaded in "+str(datetime.datetime.now()-a))
            else:
                print("***** "+YYYYMMDD + ": *_"+version+" not available")
    if success:
        print("***** "+YYYYMMDD + ": Download success *****")
        return short_file
    else:
        print("***** "+YYYYMMDD + ": Download failure *****")
        return "N/A"

"""
def download(YYYYMMDD,directory,product_family,product,version_priority): #will download rpw tds from lesia
    #product example: _rpw-tds-surv-tswf-e_
    #product example: _rpw-tds-surv-stat_
    
    #product family example: tds_stat
    #product family example: tds_wf_e
    
    #version priority exapmle: ["V02.cdf","V01.cdf"]
    
    print("***** "+YYYYMMDD + ": Download initiated *****")
    a = datetime.datetime.now()
    os.chdir(directory)
    date=str(YYYYMMDD) 
    myurl = "https://rpw.lesia.obspm.fr/roc/data/pub/solo/rpw/data/L2/"+product_family+"/"
    year = date[:4]
    month = date[4:6]
    suffix = "_V02.cdf"
    short_file = "solo_L2"+product+date+suffix
    myfile = myurl+year+"/"+month+"/"+short_file
    
    r = requests.get(myfile, allow_redirects=True)
    if not str(r)=="<Response [404]>":
        open(short_file, 'wb').write(r.content)
        print(short_file+" downlaoded")
        print(str(round(os.path.getsize(short_file)/(1024**2),ndigits=2))+" MiB dowloaded in "+str(datetime.datetime.now()-a))
    else:
        print("***** "+YYYYMMDD + ": *_V02.cdf not available, proceeding to *_V01.cdf")
        myfile = myfile[:len(myfile)-7]+"V01.cdf"
        short_file = short_file[:len(short_file)-7]+"V01.cdf"
        r = requests.get(myfile, allow_redirects=True)
        if not str(r)=="<Response [404]>":
            open(short_file, 'wb').write(r.content)
            print("***** "+short_file+" downlaoded")
            print("***** "+str(round(os.path.getsize(short_file)/(1024**2),ndigits=2))+" MiB dowloaded in "+str(datetime.datetime.now()-a))
        else:
            print("***** "+YYYYMMDD + ": Download failure *****")
            return "N/A"
    print("***** "+YYYYMMDD + ": Download success *****")
    return short_file
"""

def fetch(YYYYMMDD,directory,productfamily,product,version_priority,access_online): #will access local data or download if needed
    short_file = "N/A"
    for version in version_priority:
        try:
            f = open("solo_L2"+product+str(YYYYMMDD)+"_"+version)
        except:
            pass
            #print("cant open "+"solo_L2"+product+str(YYYYMMDD)+"_"+version)
        else:
            f.close()
            cdf_file = cdflib.CDF("solo_L2"+product+str(YYYYMMDD)+"_"+version)
            #cdf_file.cdf_info()
            print(str(YYYYMMDD)+": loaded locally as: "+"solo_L2"+product+str(YYYYMMDD)+"_"+version)
            return cdf_file
    if short_file == "N/A":
        if access_online:
            short_file = download(YYYYMMDD,directory,productfamily,product,version_priority)
    if short_file == "N/A":
        raise LookupError 
    else:
        cdf_file = cdflib.CDF(short_file)
        print(YYYYMMDD + ": Fetched "+short_file)
        return cdf_file
    
"""

#rswf
YYYYMMDD = "20210325"
waveform = fetch(YYYYMMDD,'C:\\Users\\skoci\\Disk Google\\000 Škola\\UIT\\getting data\\solo\\rpw\\tds_wf_e',"tds_wf_e","_rpw-tds-surv-rswf-e_",["V04.cdf","V03.cdf","V02.cdf","V01.cdf"],True)

#tswf
YYYYMMDD = "20210206"
waveform = fetch(YYYYMMDD,'C:\\Users\\skoci\\Disk Google\\000 Škola\\UIT\\getting data\\solo\\rpw\\tds_wf_e',"tds_wf_e","_rpw-tds-surv-tswf-e-cdag_",["V04.cdf","V03.cdf","V02.cdf","V01.cdf"],True)
waveform_quality = waveform.varget('QUALITY_FACT')
waveform_epoch = waveform.varget('EPOCH')
indices = np.arange(len(waveform_quality))[waveform_quality>65000]
waveforms = waveform.varget('WAVEFORM_DATA_VOLTAGE')

#mini plot
fig, ax = plt.subplots()
lenght = 2000
pulse = 18
sampling = np.mean(waveform.varget("SAMPLING_RATE"))
xtime = np.arange(0,np.size(waveforms[pulse,0,:]))/sampling*1000
ax.plot(xtime[:lenght],waveforms[pulse,0,:lenght],label="Channel 0")
ax.plot(xtime[:lenght],waveforms[pulse,1,:lenght],label="Channel 1")
ax.plot(xtime[:lenght],waveforms[pulse,2,:lenght],label="Channel 2")
ax.set_ylabel("Voltage [V]")
ax.set_xlabel("Time [ms]")
ax.set_ylim(-0.1,0.21)
ax.set_aspect(1000/25)
fig.suptitle( 'Dust Signal' )
fig.show()


stat = fetch(YYYYMMDD,'C:\\Users\\skoci\\Disk Google\\000 Škola\\UIT\\getting data\\solo\\rpw\\tds_stat',"tds_stat","_rpw-tds-surv-stat_",["V04.cdf","V03.cdf","V02.cdf","V01.cdf"],True)
stat_epoch = stat.varget('EPOCH')[stat.varget('DU_NR_IMPACT')==1]
stat_impacts = sum(stat.varget('DU_NR_IMPACT')[stat.varget('DU_NR_IMPACT')<2])
stat_exposure = sum(stat.varget("snapshot_len")/stat.varget("sampling_rate"))
stat_amplitudes = stat.varget('DU_MED_AMP')[stat.varget('DU_NR_IMPACT')==1]

for i in range(len(indices)):
    print((waveform_epoch[indices[i]] - stat_epoch[i])/1000000000)

for i in range(len(indices)):
    for c in [0,1,2]:
        plt.plot(waveforms[indices[i],c,:],label=c)
        plt.title("event "+str(indices[i])+"; amplitude = "+str(stat_amplitudes[i]))
    plt.legend()
    plt.show()


"""
 
