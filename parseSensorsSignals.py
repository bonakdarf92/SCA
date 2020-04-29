from tqdm import trange
import csv
import json 
import numpy as np
with open('.\\Darmstadt_verkehr\\sensor_config.json') as f:
    sensors = json.load(f)

knoten = ["A003", "A004", "A005", "A006", "A007", "A023", "A022", "A028", "A029", "A045", "A030", "A139", "A102", "A104"]
knoten_darmstadt = ["A160","A161","A075","A174","A046","A012","A059","A110","A111","A173","A037","A162","A163","A104","A023","A028",
                    "A029","A030","A031","A032","A033","A001","A002","A007","A003","A004","A005","A006","A034","A036","A035","A141",
                    "A131","A200","A168","A043","A099","A098","A022","A045","A139","A102","A081","A077","A084","A051","A044","A159",
                    "A049","A042","A182","A013","A025","A026","A071","A027","A085","A086","A041","A116","A017","A181","A019","A016",
                    "A014","A021","A100","A079","A078","A066","A057","A048","A020","A103","A015","A151","A083","A125",
                    "A067","A155","A154","A128","A137","A170","A169","A147","A142","A144","A145","A146","A136","A095",
                    "A096","A097","A010","A089","A008","A080","A105","A090","A069","A058","A070","A126","A076","A011","A140","A074",
                    "A150","A061","A150","A175","A024"]
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
    data_csv = '.\\Darmstadt_verkehr\\2019_11_18_darmstadtUI\\{}.csv'.format(k)  #19 Januar
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


np.savez('.\\Darmstadt_verkehr\\SensorData_{}'.format('Sensor_Small_18_11_2019_Counts'),new_sensors)
