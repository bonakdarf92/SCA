from tqdm import trange
import csv
import json 
import numpy as np
with open('.\\Darmstadt_verkehr\\sensor_config.json') as f:
    sensors = json.load(f)

knoten = ["A003", "A004", "A005", "A006", "A007", "A023", "A022", "A028", "A029", "A045", "A030", "A139", "A102", "A104"]
new_sensors = {item["ID"]:item for item in sensors}

for j in knoten:
    if j == "A004":
        new_sensors[j]["signals"] = np.empty((1440,43))
        new_sensors[j]["signals"][:] = np.nan
    elif j == "A007":
        new_sensors[j]["signals"] = np.empty((1440,15))
        new_sensors[j]["signals"][:] = np.nan
    elif j == "A022":
        new_sensors[j]["signals"] = np.empty((1440,39))
        new_sensors[j]["signals"][:] = np.nan
    elif j == "A028":
        new_sensors[j]["signals"] = np.empty((1440,14))
        new_sensors[j]["signals"][:] = np.nan
    elif j == "A045":
        new_sensors[j]["signals"] = np.empty((1440,46))
        new_sensors[j]["signals"][:] = np.nan
    elif j == "A030":
        new_sensors[j]["signals"] = np.empty((1440,20))
        new_sensors[j]["signals"][:] = np.nan
    elif j == "A104":
        new_sensors[j]["signals"] = np.empty((1440,35))
        new_sensors[j]["signals"][:] = np.nan
    else:
        new_sensors[j]["signals"] = np.empty((1440,len(new_sensors[j]["detectors"])))
        new_sensors[j]["signals"][:] = np.nan 

for k in trange(1440):
    data_csv = '.\\Darmstadt_verkehr\\2020_1_19_darmstadtUI\\{}.csv'.format(k)
    try:
        my_data = np.genfromtxt(data_csv,delimiter=';',dtype=object,skip_header=0,skip_footer=2,deletechars="\r")
        for j in knoten:
            if (j != "A030") and (j != "A102") and (j != "A104"):
                selec = my_data[:,2] == bytes(j.replace('0',' '),'utf-8')
            elif j == "A030":
                selec = my_data[:,2] == b"A 30" #bytes(j.replace('0',' '),'utf-8')
            elif j == "A102":
                selec = my_data[:,2] == b"A102" #bytes(j.replace('0',' '),'utf-8')
            elif j == "A104":
                selec = my_data[:,2] == b"A104" #bytes(j.replace('0',' '),'utf-8')
            new_sensors[j]["signals"][k,:] = np.array([k.decode().split('\\r')[0] for k in my_data[selec,6].astype('S')])

    except (FileNotFoundError,ValueError,OSError) as e:
        print("CSV {} nicht vorhanden oder Signal {} fehlt".format(data_csv, j))
        pass


np.savez('.\\Darmstadt_verkehr\\SensorData_{}'.format('Sensor_Small_19_01_2020_Counts'),new_sensors)