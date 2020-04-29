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
                    "A131","A200","A168","A043","A099","A098","A022","A045","A139","A102","A081","A077","A084","A051","A044","A159",
                    "A049","A042","A182","A013","A025","A026","A071","A027","A085","A086","A041","A116","A017","A181","A019","A016",
                    "A014","A021","A100","A079","A078","A066","A057","A048","A020","A103","A015","A151","A083","A125",
                    "A067","A155","A154","A128","A137","A170","A169","A147","A142","A144","A145","A146","A136","A095",
                    "A096","A097","A010","A089","A008","A080","A105","A090","A069","A058","A070","A126","A076","A011","A140","A074",
                    "A150","A061","A150","A175","A024"]
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
            new_sensors[j]["signals"] = np.empty((1440,13))
            new_sensors[j]["signals"][:] = np.nan
        elif j == "A111":
            new_sensors[j]["signals"] = np.empty((1440,7))
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
        elif j == "A140":
            new_sensors[j]["signals"] = np.empty((1440,4))
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
<<<<<<< HEAD
    data_csv = '.\\Darmstadt_verkehr\\{}_{}_{}_darmstadtUI\\{}.csv'.format(date["year"],date["month"],date["day"],k)
=======
    data_csv = '.\\Darmstadt_verkehr\\2019_11_18_darmstadtUI\\{}.csv'.format(k)  #19 Januar
>>>>>>> 891f9f779222ba29ccdce4ba18b8633008342dea
    try:
        my_data = np.genfromtxt(data_csv,delimiter=';',dtype=object,skip_header=0,skip_footer=2,deletechars="\r")
        for j in knoten:
            if ((j != "A030") and (j != "A102") and (j != "A104") and (j != "A160") and (j != "A200") and (j != "A100") and (j!="A20") 
                and (j != "A103") and (j != "A170") and (j != "A010") and (j != "A080") and (j != "A105") and (j != "A090") and (j!="A070")
                and (j != "A140") and (j != "A150")):
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
            elif j == "A200":
                selec = my_data[:,2] == b"A200"
            elif j == "A140":
                selec = my_data[:,2] == b"A140"
            elif j == "A150":
                selec = my_data[:,2] == b"A150"
            elif j == "A170":
                selec = my_data[:,2] == b"A170"
            elif j == "A103":
                selec = my_data[:,2] == b"A103"
            elif j == "A105":
                selec = my_data[:,2] == b"A105"
            elif j == "A102":
                selec = my_data[:,2] == b"A102" #bytes(j.replace('0',' '),'utf-8')
            elif j == "A104":
                selec = my_data[:,2] == b"A104" #bytes(j.replace('0',' '),'utf-8')
            new_sensors[j]["signals"][k,:] = np.array([k.decode().split('\\r')[0] for k in my_data[selec,6].astype('S')])

    except (FileNotFoundError,ValueError,OSError) as e:
        print("CSV {} nicht vorhanden oder Signal {} fehlt".format(data_csv, j))
        pass


<<<<<<< HEAD
np.savez('.\\Darmstadt_verkehr\\SensorData_Sensor_{}_{}_{}_{}_Counts'.format(flag,date["day"],date["month"],date["year"]),new_sensors)
=======
np.savez('.\\Darmstadt_verkehr\\SensorData_{}'.format('Sensor_Small_18_11_2019_Counts'),new_sensors)
>>>>>>> 891f9f779222ba29ccdce4ba18b8633008342dea
