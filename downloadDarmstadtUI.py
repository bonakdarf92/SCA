import csv
from urllib import request
import os
import tqdm
import math
import platform

url1_link = "https://darmstadt.ui-traffic.de/resources/CSVExport?from=6%2F10%2F19+4%3A59+AM&to=6%2F10%2F19+4%3A59+AM"
url2_link = "https://darmstadt.ui-traffic.de/resources/CSVExport?from=6%2F10%2F19+4%3A58+AM&to=6%2F10%2F19+4%3A58+AM"
url3_link = "https://darmstadt.ui-traffic.de/resources/CSVExport?from=6/10/19+2:59+AM&to=6/10/19+3:59+AM"


def create_url(year,month,day,duration):
    """
    This function generates the url string for download
    
    @param:\n
        year (int): desired year of observation\n
        month (int): desired month of observation\n
        day (int): desired day of observation\n
        duration (string) yet not implemented\n
    @return:\n
        out (list): strings containing all urls for download
    """
    # Ground api url
    ground_url = "https://darmstadt.ui-traffic.de/resources/CSVExport?from="

    # initialize list for return
    out = []
    for h in range(24):             # iterate over all 24 hours of the day
        for m in range(60):         # iterate over all 60 minutes of a hour
            if h < 12:              # if in the first 12 hours use AM notation
                sub = "{}/{}/{}+{}:{}+AM&to={}/{}/{}+{}:{}+AM".format(month,day,year,h,m,month,day,year,h,m)
            else:                   # else use PM notation
                sub = "{}/{}/{}+{}:{}+PM&to={}/{}/{}+{}:{}+PM".format(month,day,year,h,m,month,day,year,h,m)
            out.append(ground_url + sub)
    return out



def download_data(csv_url, dest_file, current_counter, directory):
    try:
        response = request.urlopen(csv_url,timeout=5)
    except request.URLError:
        print("Error occured during connection")
        print("Try reconnecting to server and redownload")
        try:
            response = request.urlopen(csv_url,timeout=5)
            print("Second attempt succesful")
        except:
            print("Second attempt to connect failed")
            print("Writing into missing files")
            return current_counter
         
    csv = response.read()
    csv_str = str(csv)
    lines = csv_str.split("\\n")
    dest_url = dest_file + ".csv"
    if platform.system() == 'Windows':
        #print("Windows System")
        fx = open(os.path.join('C:\\Users\\FaridLenovo\\Desktop\\SCA\\Darmstadt_verkehr', directory, dest_url),"w")
    elif platform.system() == 'Darwin':
        print("Mac OS System")
        fx = open(os.path.join('/Users/faridbonakdar/Documents/MasterThesis/SCA/Darmstadt_verkehr', directory, dest_url),"w")
    for line in lines:
        fx.write(line + "\n")
    fx.close()

def name_file(k):
    """[summary]
    
    Arguments:
        k {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    hour,minute = divmod(k,60)
    return "{}_{}".format(hour, minute)



def make_dir(name):
    current_path = os.path.join(os.getcwd(),"Darmstadt_verkehr")
    newpath = name
    try:
        os.mkdir(os.path.join(current_path,name))
    except FileExistsError:
        return name
    print("Ordner erstellt")
    return name
    #pass 

m = 11
day = 15
date = {'year':2019,'month':m,'day':day}
test = create_url(date['year'], date['month'], date['day'], None)
missing_files = []

d = make_dir("{}_{}_{}_darmstadtUI".format(date['year'],date['month'],date['day']))
#make_dir(os.path.)

#for k in range(1440):
#    print(name_file(k))
print("Jahr 2019, Monat {}, Tag {}".format(m,day))

# TODO check for download errors and path saving
for k in tqdm.trange(len(test)):
    url = test[k]
    #print("Datei {} - aktueller Stand: {}".format(url,round(k/len(test),2)))
    missing = download_data(url,str(k), k, d)
    missing_files.append(missing)


"""
# TODO test
if missing_files:
    for miss in missing_files:
        url = missing_files[miss]
        print("Redownload missing file: {}", url)
        download_data(url,)
"""
#download_data(url1_link,str(3))
print("ok")
