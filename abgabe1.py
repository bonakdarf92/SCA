from DarmstadtNetwork import DarmstadtNetwork
geo = dict(north=49.874,south=49.8679,west=8.6338,east=8.6517)
D_city = DarmstadtNetwork(geo,"Abgabe")
D_city.load_darmstadt(show=True)
D_city.plot_map()

#D_city.download_darmstadt(D_city.geo, D_city.nameFile,loc=".")
#D_city = DarmstadtNetwork.load_darmstadt(DarmstadtNetwork, name="Abgabe",show=True)