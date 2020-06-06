from tqdm import trange
import csv
import json 
import numpy as np
with open('.\\Darmstadt_verkehr\\sensor_config.json') as f:
    sensors = json.load(f)

flag = "Big"
knoten = ["A003", "A004", "A005", "A006", "A007", "A023", "A022", "A028", "A029", "A045", "A030", "A139", "A102", "A104"]
knoten_darmstadt = ["A160","A161","A075","A174","A046","A012","A059","A110","A111","A173","A037","A162","A163","A104","A023","A028",
                    "A029","A030","A031","A032","A033","A001","A002","A007","A003","A004","A005","A006","A034","A036","A035","A141",
                    "A131","A168","A043","A099","A098","A022","A045","A102","A081","A077","A084","A051","A159","A009",
                    "A049","A042","A013","A026","A071","A027","A085","A086","A041","A017","A019","A016",
                    "A014","A021","A100","A066","A057","A048","A020","A103","A015","A151","A083","A134",
                    "A067","A155","A154","A128","A137","A170","A169","A147","A142","A144","A146","A136","A095",
                    "A096","A097","A010","A008","A080","A090","A069","A070","A126","A076","A011",
                    "A150","A061","A150","A024"]
new_sensors = {item["ID"]:item for item in sensors}

if flag == "Small":
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
            print("Else Block")
            new_sensors[j]["signals"] = np.empty((1440,len(new_sensors[j]["detectors"])))
            new_sensors[j]["signals"][:] = np.nan 

if flag == "Big":
    for j in knoten_darmstadt:
        if j == "A004":
            new_sensors[j]["signals"] = np.empty((1440,43))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A160":
            new_sensors[j]["signals"] = np.empty((1440,17))
            new_sensors[j]["signals"][:] = np.nan 
        elif j == "A161":
            new_sensors[j]["signals"] = np.empty((1440,26))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A075":
            new_sensors[j]["signals"] = np.empty((1440,46))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A174":
            new_sensors[j]["signals"] = np.empty((1440,15))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A009":
            new_sensors[j]["signals"] = np.empty((1440,17))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A076":
            new_sensors[j]["signals"] = np.empty((1440,31))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A134":
            new_sensors[j]["signals"] = np.empty((1440,11))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A126":
            new_sensors[j]["signals"] = np.empty((1440,26))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A070":
            new_sensors[j]["signals"] = np.empty((1440,16))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A080":
            new_sensors[j]["signals"] = np.empty((1440,9))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A066":
            new_sensors[j]["signals"] = np.empty((1440,16))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A168":
            new_sensors[j]["signals"] = np.empty((1440,23))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A013":
            new_sensors[j]["signals"] = np.empty((1440,32))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A041":
            new_sensors[j]["signals"] = np.empty((1440,31))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A128":
            new_sensors[j]["signals"] = np.empty((1440,16))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A142":
            new_sensors[j]["signals"] = np.empty((1440,29))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A136":
            new_sensors[j]["signals"] = np.empty((1440,12))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A096":
            new_sensors[j]["signals"] = np.empty((1440,10))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A046":
            new_sensors[j]["signals"] = np.empty((1440,28))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A012":
            new_sensors[j]["signals"] = np.empty((1440,22))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A059":
            new_sensors[j]["signals"] = np.empty((1440,24))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A110":                                                   # abfangen
            new_sensors[j]["signals"] = np.empty((1440,14))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A111":
            new_sensors[j]["signals"] = np.empty((1440,7))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A081":
            new_sensors[j]["signals"] = np.empty((1440,28))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A035":
            new_sensors[j]["signals"] = np.empty((1440,14))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A036":
            new_sensors[j]["signals"] = np.empty((1440,31))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A173":
            new_sensors[j]["signals"] = np.empty((1440,11))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A037":
            new_sensors[j]["signals"] = np.empty((1440,29))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A162":
            new_sensors[j]["signals"] = np.empty((1440,28))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A163":
            new_sensors[j]["signals"] = np.empty((1440,20))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A104":                                                   # abfangen
            new_sensors[j]["signals"] = np.empty((1440,35))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A023":
            new_sensors[j]["signals"] = np.empty((1440,23))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A028":
            new_sensors[j]["signals"] = np.empty((1440,14))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A029":
            new_sensors[j]["signals"] = np.empty((1440,2))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A030":
            new_sensors[j]["signals"] = np.empty((1440,20))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A031":
            new_sensors[j]["signals"] = np.empty((1440,4))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A032":
            new_sensors[j]["signals"] = np.empty((1440,24))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A033":
            new_sensors[j]["signals"] = np.empty((1440,37))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A001":
            new_sensors[j]["signals"] = np.empty((1440,25))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A002":
            new_sensors[j]["signals"] = np.empty((1440,26))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A007":
            new_sensors[j]["signals"] = np.empty((1440,15))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A003":
            new_sensors[j]["signals"] = np.empty((1440,31))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A034":
            new_sensors[j]["signals"] = np.empty((1440,8))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A141":
            new_sensors[j]["signals"] = np.empty((1440,24))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A099":
            new_sensors[j]["signals"] = np.empty((1440,16))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A098":
            new_sensors[j]["signals"] = np.empty((1440,36))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A022":
            new_sensors[j]["signals"] = np.empty((1440,39))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A045":
            new_sensors[j]["signals"] = np.empty((1440,46))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A077":
            new_sensors[j]["signals"] = np.empty((1440,10))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A044":
            new_sensors[j]["signals"] = np.empty((1440,4))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A071":
            new_sensors[j]["signals"] = np.empty((1440,22))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A027":
            new_sensors[j]["signals"] = np.empty((1440,41))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A086":
            new_sensors[j]["signals"] = np.empty((1440,50))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A016":
            new_sensors[j]["signals"] = np.empty((1440,12))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A014":
            new_sensors[j]["signals"] = np.empty((1440,27))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A021":
            new_sensors[j]["signals"] = np.empty((1440,35))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A057":
            new_sensors[j]["signals"] = np.empty((1440,46))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A048":
            new_sensors[j]["signals"] = np.empty((1440,12))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A020":
            new_sensors[j]["signals"] = np.empty((1440,47))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A103":
            new_sensors[j]["signals"] = np.empty((1440,12))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A015":
            new_sensors[j]["signals"] = np.empty((1440,49))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A151":
            new_sensors[j]["signals"] = np.empty((1440,7))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A067":
            new_sensors[j]["signals"] = np.empty((1440,28))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A155":
            new_sensors[j]["signals"] = np.empty((1440,28))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A154":
            new_sensors[j]["signals"] = np.empty((1440,19))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A137":
            new_sensors[j]["signals"] = np.empty((1440,55))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A170":
            new_sensors[j]["signals"] = np.empty((1440,24))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A084":
            new_sensors[j]["signals"] = np.empty((1440,4))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A051":
            new_sensors[j]["signals"] = np.empty((1440,32))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A159":
            new_sensors[j]["signals"] = np.empty((1440,20))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A049":
            new_sensors[j]["signals"] = np.empty((1440,37))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A042":
            new_sensors[j]["signals"] = np.empty((1440,15))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A010":
            new_sensors[j]["signals"] = np.empty((1440,14))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A169":
            new_sensors[j]["signals"] = np.empty((1440,6))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A147":
            new_sensors[j]["signals"] = np.empty((1440,55))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A144":
            new_sensors[j]["signals"] = np.empty((1440,21))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A146":
            new_sensors[j]["signals"] = np.empty((1440,25))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A095":
            new_sensors[j]["signals"] = np.empty((1440,28))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A097":
            new_sensors[j]["signals"] = np.empty((1440,36))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A090":
            new_sensors[j]["signals"] = np.empty((1440,21))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A069":
            new_sensors[j]["signals"] = np.empty((1440,46))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A150":
            new_sensors[j]["signals"] = np.empty((1440,5))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A061":
            new_sensors[j]["signals"] = np.empty((1440,12))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A024":
            new_sensors[j]["signals"] = np.empty((1440,21))
            new_sensors[j]["signals"][:] = np.nan
        else:
            new_sensors[j]["signals"] = np.empty((1440,len(new_sensors[j]["detectors"])))
            new_sensors[j]["signals"][:] = np.nan 

date = {'year':2020,'month':1,'day':16}

for k in trange(1440):
    data_csv = '.\\Darmstadt_verkehr\\{}_{}_{}_darmstadtUI\\{}.csv'.format(date["year"],date["month"],date["day"],k)
    try:
        my_data = np.genfromtxt(data_csv,delimiter=';',dtype=object,skip_header=0,skip_footer=2,deletechars="\r")
        for j in knoten_darmstadt:
            if ((j != "A030") and (j != "A102") and (j != "A104") and (j != "A160") and (j != "A100") and (j!="A020") 
                and (j != "A103") and (j != "A170") and (j != "A010") and (j != "A080") and (j != "A090") and (j!="A070") and (j != "A150") and (j != "A110")):
                selec = my_data[:,2] == bytes(j.replace('0',' '),'utf-8')
            elif j == "A030":
                selec = my_data[:,2] == b"A 30" #bytes(j.replace('0',' '),'utf-8')
            elif j == "A110":
                selec = my_data[:,2] == b"A110"
            elif j == "A160":
                selec = my_data[:,2] == b"A160"
            elif j == "A100":
                selec = my_data[:,2] == b"A100"
            elif j == "A010":
                selec = my_data[:,2] == b"A 10"
            elif j == "A020":
                selec = my_data[:,2] == b"A 20"
            elif j == "A070":
                selec = my_data[:,2] == b"A 70"
            elif j == "A080":
                selec = my_data[:,2] == b"A 80"
            elif j == "A090":
                selec = my_data[:,2] == b"A 90"
            elif j == "A150":
                selec = my_data[:,2] == b"A150"
            elif j == "A170":
                selec = my_data[:,2] == b"A170"
            elif j == "A103":
                selec = my_data[:,2] == b"A103"
            elif j == "A102":
                selec = my_data[:,2] == b"A102"
            elif j == "A104":
                selec = my_data[:,2] == b"A104"
            #else:
            #    selec = 
            new_sensors[j]["signals"][k,:] = np.array([k.decode().split('\\r')[0] for k in my_data[selec,6].astype('S')])

    except (FileNotFoundError,ValueError,OSError) as e:
        print("CSV {} nicht vorhanden oder Signal {} fehlt".format(data_csv, j))
        print(e)
        pass


np.savez('.\\Darmstadt_verkehr\\SensorData_Sensor_{}_{}_{}_{}_Counts'.format(flag,date["day"],date["month"],date["year"]),new_sensors)
