import numpy as np
import csv

def load_ephemeris(spacecraft):

    time = np.zeros(0)
    hae_r = np.zeros(0)
    hae_v = np.zeros(0)

    if spacecraft == "solo":
        ephemeris_file = "solo_ephemeris_noheader.txt"
    else:
        raise LookupError("Unknown spacecraft")
        
    with open(ephemeris_file) as file:  
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            time = np.append(time,float(row[0]))
            hae_r = np.append(hae_r,[float(row[2]),float(row[3]),float(row[4])])
            hae_v = np.append(hae_v,[float(row[5]),float(row[6]),float(row[7])])
        hae_r = np.reshape(hae_r,((len(hae_r)//3,3)))
        hae_v = np.reshape(hae_v,((len(hae_v)//3,3)))
        hae_phi = np.degrees(np.arctan2(hae_r[:,1],hae_r[:,0]))

    #compute radial and tangential velocities
    radial_v = np.zeros(len(hae_r[:,0]))
    tangential_v = np.zeros(len(hae_r[:,0]))
    for i in range(len(hae_r[:,0])):
        unit_radial = hae_r[i,:]/np.linalg.norm(hae_r[i,:])
        radial_v[i] = np.inner(unit_radial,hae_v[i,:])
        tangential_v[i] = np.linalg.norm(hae_v[i,:]-radial_v[i]*unit_radial)
        
    return time, hae_r, hae_v, hae_phi, radial_v, tangential_v