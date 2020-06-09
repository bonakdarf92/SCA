import networkx as nx 
import matplotlib.pyplot as plt 
import numpy as np
plt.set_cmap('rainbow')
plt.rcParams.update({'font.size':18})
Darmstadt = nx.DiGraph()
# A137
Darmstadt.add_node("137_D1115", pos=[30,2000],signal=23,starter=True)
Darmstadt.add_node("137_D1114", pos=[29.5,1997],signal=22,starter=True)
Darmstadt.add_node("137_D1113", pos=[29,1994],signal=21,starter=True)
Darmstadt.add_node("137_D1112", pos=[28.5,1991],signal=20,starter=True)
Darmstadt.add_node("137_D1111", pos=[28,1988],signal=19,starter=True)
Darmstadt.add_node("137_D912",  pos=[36,1970],signal=18)
Darmstadt.add_node("137_D911",  pos=[38,1970],signal=17)
Darmstadt.add_node("137_D71",   pos=[40,1970],signal=16)
Darmstadt.add_node("137_D122",  pos=[49.5,1985],signal=15)
Darmstadt.add_node("137_D121",  pos=[49.4,1982.5],signal=14)
Darmstadt.add_node("137_D112",  pos=[49.3,1980],signal=13)
Darmstadt.add_node("137_D111",  pos=[49.2,1977.5],signal=12)
Darmstadt.add_node("137_D102",  pos=[49.1,1975],signal=11)
Darmstadt.add_node("137_D101",  pos=[49,1972.5],signal=10)
Darmstadt.add_node("137_D511",  pos=[45,2006],signal=33,ende=True)
Darmstadt.add_node("137_D512",  pos=[44.7,2003],signal=34,ende=True)
Darmstadt.add_node("137_D61",   pos=[44.4,2000],signal=35,ende=True)
Darmstadt.add_node("137_D92",   pos=[57,1965],signal=9)
Darmstadt.add_node("137_D91",   pos=[59,1965],signal=8)
Darmstadt.add_node("137_D81",   pos=[61,1965],signal=7)
Darmstadt.add_node("137_D11",   pos=[61.5,2015],signal=1,starter=True)
Darmstadt.add_node("137_D21",   pos=[59,2006],signal=2,starter=True)
Darmstadt.add_node("137_D22",   pos=[61,2006],signal=3,starter=True)
Darmstadt.add_node("137_D51",   pos=[68.5,1988],signal=5)
Darmstadt.add_node("137_D52",   pos=[68,1985],signal=6)
Darmstadt.add_node("137_D41",   pos=[69.5,1991],signal=4)
Darmstadt.add_node("137_E1",    pos=[33,2005],ende=True)
Darmstadt.add_node("137_E2",    pos=[34,1965],ende=True)
Darmstadt.add_node("137_E3",    pos=[53,1960])
Darmstadt.add_node("137_E4",    pos=[68,1977.5],ende=True)
Darmstadt.add_node("137_E5",    pos=[64,2008])
Darmstadt.add_edge("137_D1115","137_D122",weight=1)
Darmstadt.add_edge("137_D1115","137_D121",weight=1)
Darmstadt.add_edge("137_D1114","137_D112",weight=1)
Darmstadt.add_edge("137_D1113","137_D111",weight=1)
Darmstadt.add_edge("137_D1112","137_D102",weight=1)
Darmstadt.add_edge("137_D1111","137_D101",weight=1)
Darmstadt.add_edge("137_D1111","137_E2",weight=1)
Darmstadt.add_edge("137_D122","137_E5",weight=1)
Darmstadt.add_edge("137_D121","137_E5",weight=1)
Darmstadt.add_edge("137_D112","137_E4",weight=1)
Darmstadt.add_edge("137_D111","137_E4",weight=1)
Darmstadt.add_edge("137_D102","137_E3",weight=1)
Darmstadt.add_edge("137_D101","137_E3",weight=1)
Darmstadt.add_edge("137_D92","137_D512",weight=1)
Darmstadt.add_edge("137_D91","137_D511",weight=1)
Darmstadt.add_edge("137_D81","137_E4",weight=1)
Darmstadt.add_edge("137_D81","137_E5",weight=1)
Darmstadt.add_edge("137_D51","137_D511",weight=1)
Darmstadt.add_edge("137_D52","137_D512",weight=1)
Darmstadt.add_edge("137_D52","137_D61",weight=1)
Darmstadt.add_edge("137_D41","137_E5",weight=1)
Darmstadt.add_edge("137_D11","137_E1",weight=1)
Darmstadt.add_edge("137_D21","137_D512",weight=1)
Darmstadt.add_edge("137_D21","137_D61",weight=1)
Darmstadt.add_edge("137_D22","137_E3",weight=1)
Darmstadt.add_edge("137_D912","137_E1",weight=1)
Darmstadt.add_edge("137_D911","137_E1",weight=1)
Darmstadt.add_edge("137_D71","137_D111",weight=1)
Darmstadt.add_edge("137_D71","137_D101",weight=1)

# 170
Darmstadt.add_node("170_D111",pos=[368,1977.5],signal=11)
Darmstadt.add_node("170_D61.1",pos=[374,1978.5],signal=3)
Darmstadt.add_node("170_D51",pos=[374,1979],signal=1)
Darmstadt.add_node("170_D71",pos=[372,1975],signal=6)#starter = True
Darmstadt.add_node("170_D91.1",pos=[371,1975],signal=8)#starter=True
Darmstadt.add_node("170_E1",pos=[368,1979])
Darmstadt.add_node("170_E2",pos=[370,1975]) # ende=True
Darmstadt.add_node("170_E3",pos=[376,1977.5])
Darmstadt.add_edge("137_E4","170_D111",weight=1)
Darmstadt.add_edge("170_E1","137_D51",weight=1)
Darmstadt.add_edge("170_E1","137_D52",weight=1)
Darmstadt.add_edge("170_E1","137_D41",weight=1)
Darmstadt.add_edge("170_D111","170_E2",weight=1)
Darmstadt.add_edge("170_D111","170_E3",weight=1)
Darmstadt.add_edge("170_D51","170_E1",weight=1)
Darmstadt.add_edge("170_D61.1","170_E2",weight=1)
Darmstadt.add_edge("170_D91.1","170_E1",weight=1)
Darmstadt.add_edge("170_D71","170_E3",weight=1)
#Darmstadt.add_edge("170_E2","136",weight=1)


#169
Darmstadt.add_node("169_D111",pos=[590,1990],signal=6)
Darmstadt.add_node("169_D51",pos=[610,1992],signal=5)
Darmstadt.add_edge("170_E3","169_D111",weight=1)
Darmstadt.add_edge("169_D51","170_D51",weight=1)
Darmstadt.add_edge("169_D51","170_D61.1",weight=1)

#147
Darmstadt.add_node("147_D121",pos=[825,2012.5],signal=25)
Darmstadt.add_node("147_D112",pos=[825,2011.5],signal=22)
Darmstadt.add_node("147_D111",pos=[825,2010.5],signal=21)
Darmstadt.add_node("147_D91.1",pos=[840,2001.5],signal=18,starter=True)
Darmstadt.add_node("147_D82",pos=[841,2001.5],signal=15,starter=True)
Darmstadt.add_node("147_D81",pos=[842,2001.5],signal=14,starter=True)
Darmstadt.add_node("147_D61",pos=[856,2019.5],signal=12)
Darmstadt.add_node("147_D53",pos=[856,2020.5],signal=8)
Darmstadt.add_node("147_D52",pos=[856,2021.5],signal=7)
Darmstadt.add_node("147_D51",pos=[856,2022.5],signal=6)
Darmstadt.add_node("147_D31.1",pos=[839.5,2030],signal=3,starter=True)
Darmstadt.add_node("147_D21",pos=[838.5,2030],signal=1,starter=True)
Darmstadt.add_node("147_E1",pos=[841,2030],ende=True)
Darmstadt.add_node("147_E2",pos=[858,2016.5])
Darmstadt.add_node("147_E3",pos=[838,1999.5],ende=True)
Darmstadt.add_edge("147_D52","169_D51",weight=1)
Darmstadt.add_edge("147_D53","169_D51",weight=1)
Darmstadt.add_edge("147_D51","147_E1",weight=1)
Darmstadt.add_edge("147_D61","147_E3",weight=1)
Darmstadt.add_edge("147_D81","147_E2",weight=1)
Darmstadt.add_edge("147_D82","147_E1",weight=1)
Darmstadt.add_edge("147_D91.1","169_D51",weight=1)
Darmstadt.add_edge("169_D111","147_D121",weight=1)
Darmstadt.add_edge("169_D111","147_D112",weight=1)
Darmstadt.add_edge("169_D111","147_D111",weight=1)
Darmstadt.add_edge("147_D21","169_D51",weight=1)
Darmstadt.add_edge("147_D21","147_E3",weight=1)
Darmstadt.add_edge("147_D31.1","147_E2",weight=1)
Darmstadt.add_edge("147_D112","147_E2",weight=1)
Darmstadt.add_edge("147_D111","147_E3",weight=1)
Darmstadt.add_edge("147_D121","147_E1",weight=1)

#136
Darmstadt.add_node("136_D41",pos=[240,1737],signal=4)
Darmstadt.add_node("136_D42",pos=[248,1731],signal=5)
Darmstadt.add_node("136_D11",pos=[267,1730.5],signal=1)
Darmstadt.add_node("136_D12",pos=[271,1733.5],signal=2)
Darmstadt.add_node("136_D21",pos=[261,1699],signal=3)
Darmstadt.add_node("136_E1",pos=[248,1733.5])
Darmstadt.add_node("136_E2",pos=[273,1733.5])
Darmstadt.add_node("136_E3",pos=[258,1699]) 
Darmstadt.add_edge("137_E3","136_D41",weight=1)
Darmstadt.add_edge("137_E3","136_D42",weight=1)
Darmstadt.add_edge("136_E1","137_D92",weight=1)
Darmstadt.add_edge("136_E1","137_D91",weight=1)
Darmstadt.add_edge("136_E1","137_D81",weight=1)
Darmstadt.add_edge("170_E2","136_D12",weight=1)
Darmstadt.add_edge("170_E2","136_D11",weight=1)
Darmstadt.add_edge("136_D21","136_E1",weight=1)
Darmstadt.add_edge("136_D11","136_E1",weight=1)
Darmstadt.add_edge("136_D12","136_E3",weight=1)
Darmstadt.add_edge("136_D41","136_E3",weight=1)
Darmstadt.add_edge("136_D42","136_E2",weight=1)
Darmstadt.add_edge("136_E3","95_D11",weight=1)
Darmstadt.add_edge("136_E3","95_D12",weight=1)
Darmstadt.add_edge("136_E2","170_D71",weight=1)
Darmstadt.add_edge("136_E2","170_D91.1",weight=1)

#142
Darmstadt.add_node("142_D41",pos=[1215,2125],signal=12)
Darmstadt.add_node("142_D42",pos=[1215.5,2126.5],signal=13)
Darmstadt.add_node("142_D43",pos=[1216,2128],signal=14)
Darmstadt.add_node("142_D44",pos=[1216.5,2129.5],signal=15)
Darmstadt.add_node("142_D33",pos=[1233,2120.5],signal=11)
Darmstadt.add_node("142_D32",pos=[1233,2109],signal=10)
Darmstadt.add_node("142_D31",pos=[1234.5,2109],signal=9)
Darmstadt.add_node("142_D24",pos=[1247.5,2135.5],signal=8)
Darmstadt.add_node("142_D23",pos=[1247.5,2137],signal=7)
Darmstadt.add_node("142_D22",pos=[1247.5,2138.5],signal=6)
Darmstadt.add_node("142_D21",pos=[1247.5,2140],signal=5)
Darmstadt.add_node("142_D11",pos=[1224.5,2155.5],signal=1,starter=True)
Darmstadt.add_node("142_D12",pos=[1226,2155],signal=2,starter=True)
Darmstadt.add_node("142_D13",pos=[1227.5,2154.5],signal=3,starter=True)
Darmstadt.add_node("142_D15",pos=[1227.5,2144],signal=4,starter=True)
Darmstadt.add_node("142_E1",pos=[1216.5,2132])
Darmstadt.add_node("142_E2",pos=[1237,2144],ende=True)
Darmstadt.add_node("142_E3",pos=[1247.5,2132.5])
Darmstadt.add_node("142_E4",pos=[1223,2120.5])
Darmstadt.add_edge("142_E1","147_D51",weight=1)
Darmstadt.add_edge("142_E1","147_D52",weight=1)
Darmstadt.add_edge("142_E1","147_D53",weight=1)
Darmstadt.add_edge("142_E1","147_D61",weight=1)
Darmstadt.add_edge("147_E2","142_D41",weight=1)
Darmstadt.add_edge("147_E2","142_D42",weight=1)
Darmstadt.add_edge("147_E2","142_D43",weight=1)
Darmstadt.add_edge("147_E2","142_D44",weight=1)
Darmstadt.add_edge("142_D11","142_E1",weight=1)
Darmstadt.add_edge("142_D21","142_E1",weight=1)
Darmstadt.add_edge("142_D22","142_E1",weight=1)
Darmstadt.add_edge("142_D23","142_E4",weight=1)
Darmstadt.add_edge("142_D24","142_E4",weight=1)
Darmstadt.add_edge("142_D15","142_E3",weight=1)
Darmstadt.add_edge("142_D33","142_E1",weight=1)
Darmstadt.add_edge("142_D32","142_E2",weight=1)
Darmstadt.add_edge("142_D31","142_E2",weight=1)
Darmstadt.add_edge("142_D12","142_E4",weight=1)
Darmstadt.add_edge("142_D13","142_E4",weight=1)
Darmstadt.add_edge("142_D41","142_E3",weight=1)
Darmstadt.add_edge("142_D42","142_E3",weight=1)
Darmstadt.add_edge("142_D43","142_E2",weight=1)
Darmstadt.add_edge("142_D44","142_E2",weight=1)
Darmstadt.add_edge("142_E3","144_D41",weight=1)
Darmstadt.add_edge("142_E3","144_D42",weight=1)

#80
Darmstadt.add_node("80_D11",pos=[1149,1754],signal=1)
Darmstadt.add_edge("80_D11","8_D11",weight=1)

#8
Darmstadt.add_node("8_D11",pos=[1137,1697],signal=1)
Darmstadt.add_node("8_D12",pos=[1134,1679],signal=2)
Darmstadt.add_node("8_D13",pos=[1136.5,1679],signal=3)
Darmstadt.add_node("8_D51",pos=[1154.5,1670],signal=12)
Darmstadt.add_node("8_D21",pos=[1154,1665.5],signal=4)
Darmstadt.add_node("8_D22",pos=[1154,1664],signal=5)
Darmstadt.add_node("8_D44",pos=[1120,1646.5],sigal=11)
Darmstadt.add_node("8_D43",pos=[1119.5,1645],sigal=10)
Darmstadt.add_node("8_D42",pos=[1119,1643.5],sigal=9)
Darmstadt.add_node("8_D41",pos=[1118.5,1641],sigal=8)
Darmstadt.add_node("8_D32",pos=[1134,1615],signal=7)
Darmstadt.add_node("8_D31",pos=[1135.5,1615],signal=6)
Darmstadt.add_node("8_E1",pos=[1146.5,1679])
Darmstadt.add_node("8_E2",pos=[1154,1651.5])
Darmstadt.add_node("8_E3",pos=[1128,1615])
Darmstadt.add_node("8_E4",pos=[1106,1646.5])
Darmstadt.add_edge("8_D44","8_E1",weight=1)
Darmstadt.add_edge("8_D43","8_E1",weight=1)
Darmstadt.add_edge("8_D42","8_E2",weight=1)
Darmstadt.add_edge("8_D41","8_E3",weight=1)
Darmstadt.add_edge("8_D32","8_E1",weight=1)
Darmstadt.add_edge("8_D31","8_E1",weight=1)
Darmstadt.add_edge("8_D31","8_E2",weight=1)
Darmstadt.add_edge("8_D22","8_E4",weight=1)
Darmstadt.add_edge("8_D21","8_E4",weight=1)
Darmstadt.add_edge("8_D51","8_E1",weight=1)
Darmstadt.add_edge("8_D12","8_E3",weight=1)
Darmstadt.add_edge("8_D13","8_E2",weight=1)
Darmstadt.add_edge("8_D11","8_E4",weight=1)
Darmstadt.add_edge("8_D11","8_D12",weight=1)
Darmstadt.add_edge("142_E4","8_D11",weight=1)
Darmstadt.add_edge("142_E4","8_D13",weight=1)
Darmstadt.add_edge("8_E1","80_D11",weight=1)
Darmstadt.add_edge("8_E1","142_D33",weight=1)
Darmstadt.add_edge("8_E1","142_D32",weight=1)
Darmstadt.add_edge("8_E1","142_D31",weight=1)
Darmstadt.add_edge("8_E4","10_D11",weight=1)
Darmstadt.add_edge("8_E4","10_D12",weigth=1)
Darmstadt.add_edge("8_E4","10_D13",weight=1)
Darmstadt.add_edge("8_E4","10_D14",weight=1)
Darmstadt.add_edge("8_E3","9_D11",weight=1)
Darmstadt.add_edge("8_E2","90_V112",weight=1)
Darmstadt.add_edge("8_E2","90_V111",weight=1)

# 10
Darmstadt.add_node("10_D11",pos=[835,1481],signal=1)
Darmstadt.add_node("10_D12",pos=[841.5,1486],signal=2)
Darmstadt.add_node("10_D13",pos=[843,1484.5],signal=3)
Darmstadt.add_node("10_D14",pos=[842,1472],signal=4)
Darmstadt.add_node("10_D21",pos=[849.5,1458],signal=5)
Darmstadt.add_node("10_D22",pos=[849,1456.5],signal=6)
Darmstadt.add_node("10_D31",pos=[807,1431.5],signal=7,starter=True)
Darmstadt.add_node("10_D32",pos=[805.5,1432.5],signal=8,starter=True)
Darmstadt.add_node("10_D33",pos=[808.5,1440.5],signal=9,starter=True)
Darmstadt.add_node("10_D41",pos=[808.5,1463],signal=10)
Darmstadt.add_node("10_D42",pos=[809,1464.5],signal=11)
Darmstadt.add_node("10_D43",pos=[797,1463.5]) #signal 16
Darmstadt.add_node("10_E1",pos=[846,1472])
Darmstadt.add_node("10_E2",pos=[849,1455])
Darmstadt.add_node("10_E3",pos=[805,1440.5],ende=True)
Darmstadt.add_node("10_E4",pos=[809,1466])
Darmstadt.add_edge("10_D11","10_E4",weight=1)
Darmstadt.add_edge("10_D12","10_E3",weight=1)
Darmstadt.add_edge("10_D13","10_E3",weight=1)
Darmstadt.add_edge("10_D14","10_E2",weight=1)
Darmstadt.add_edge("10_D21","10_E1",weight=1)
Darmstadt.add_edge("10_D21","10_E4",weight=1)
Darmstadt.add_edge("10_D22","10_E3",weight=1)
Darmstadt.add_edge("10_D31","10_E1",weight=1)
Darmstadt.add_edge("10_D31","10_E2",weight=1)
Darmstadt.add_edge("10_D32","10_E1",weight=1)
Darmstadt.add_edge("10_D33","10_E4",weight=1)
Darmstadt.add_edge("10_D41","10_E1",weight=1)
Darmstadt.add_edge("10_D42","10_E1",weight=1)
Darmstadt.add_edge("10_D43","10_E3",weight=1)
Darmstadt.add_edge("10_E1","8_D41",weight=1)
Darmstadt.add_edge("10_E1","8_D42",weight=1)
Darmstadt.add_edge("10_E1","8_D43",weight=1)
Darmstadt.add_edge("10_E1","8_D44",weight=1)
Darmstadt.add_edge("10_E4","97_D51",weight=1)
Darmstadt.add_edge("10_E4","97_D52",weight=1)
Darmstadt.add_edge("10_E4","97_D61.1",weight=1)
Darmstadt.add_edge("10_E2","134_A4",weight=1)

# 97
Darmstadt.add_node("97_D51",pos=[700,1526],signal=4)
Darmstadt.add_node("97_D52",pos=[700,1524.5],signal=5)
Darmstadt.add_node("97_D61.1",pos=[700,1523],signal=8)
Darmstadt.add_node("97_D81",pos=[687,1512],signal=12,starter=True)
Darmstadt.add_node("97_D82.1",pos=[685,1512.5],signal=13,starter=True)
Darmstadt.add_node("97_D111",pos=[666,1524.5],signal=17)
Darmstadt.add_node("97_D112",pos=[667,1527],signal=18)
Darmstadt.add_node("97_D121.1",pos=[668,1529],signal=22)
Darmstadt.add_node("97_D21.1",pos=[682,1539.5],signal=1,starter=True)
Darmstadt.add_node("97_E1",pos=[684.5,1539.5],ende=True)
Darmstadt.add_node("97_E2",pos=[700,1526.5])
Darmstadt.add_node("97_E3",pos=[682.5,1512.5],ende=True)
Darmstadt.add_node("97_E4",pos=[668,1532])
Darmstadt.add_edge("97_D51","97_E4",weight=1)
Darmstadt.add_edge("97_D51","97_E1",weight=1)
Darmstadt.add_edge("97_D52","97_E4",weight=1)
Darmstadt.add_edge("97_D61.1","97_E3",weight=1)
Darmstadt.add_edge("97_D81","97_E2",weight=1)
Darmstadt.add_edge("97_D81","97_E1",weight=1)
Darmstadt.add_edge("97_D82.1","97_E4",weight=1)
Darmstadt.add_edge("97_D111","97_E3",weight=1)
Darmstadt.add_edge("97_D111","97_E2",weight=1)
Darmstadt.add_edge("97_D112","97_E2",weight=1)
Darmstadt.add_edge("97_D121.1","97_E1",weight=1)
Darmstadt.add_edge("97_D21.1","97_E2",weight=1)
Darmstadt.add_edge("97_D21.1","97_E3",weight=1)
Darmstadt.add_edge("97_D21.1","97_E4",weight=1)
Darmstadt.add_edge("97_E2","10_D41",weight=1)
Darmstadt.add_edge("97_E2","10_D42",weight=1)
Darmstadt.add_edge("97_E2","10_D43",weight=1)
Darmstadt.add_edge("97_E4","96_D21",weight=1)
Darmstadt.add_edge("97_E4","96_D22",weight=1)

# 96
Darmstadt.add_node("96_D21",pos=[488,1581.5],signal=1)
Darmstadt.add_node("96_D22",pos=[487,1579.5],signal=2)
Darmstadt.add_node("96_D31",pos=[459.5,1572.5],signal=3,starter=True)
Darmstadt.add_node("96_D32",pos=[457.5,1573.5],signal=4,starter=True)
Darmstadt.add_node("96_D33",pos=[455.5,1574.5],signal=5,starter=True)
Darmstadt.add_node("96_D41",pos=[424.5,1590],signal=6)
Darmstadt.add_node("96_D42",pos=[425.5,1592],signal=7)
Darmstadt.add_node("96_E2",pos=[487,1576])
Darmstadt.add_edge("96_D42","96_E2",weight=1)
Darmstadt.add_edge("96_D41","96_E2",weight=1)
Darmstadt.add_edge("96_D31","96_E2",weight=1)
Darmstadt.add_edge("96_E2","97_D111",weight=1)
Darmstadt.add_edge("96_E2","97_D112",weight=1)
Darmstadt.add_edge("96_E2","97_D121.1",weight=1)
Darmstadt.add_edge("96_D21","95_D20",weight=1)
Darmstadt.add_edge("96_D22","95_D21",weight=1)
Darmstadt.add_edge("96_D22","95_D22",weight=1)
Darmstadt.add_edge("96_D33","95_D21",weight=1)
Darmstadt.add_edge("96_D33","95_D22",weight=1)
Darmstadt.add_edge("96_D32","95_D20",weight=1)
Darmstadt.add_edge("96_D32","95_D21",weight=1)

# 95
Darmstadt.add_node("95_D11",pos=[323.5,1642],signal=1)
Darmstadt.add_node("95_D12",pos=[325.5,1643.5],signal=2)
Darmstadt.add_node("95_D20",pos=[364.5,1615],signal=3)
Darmstadt.add_node("95_D21",pos=[364,1613],signal=4)
Darmstadt.add_node("95_D22",pos=[349,1616.5],signal=5)
Darmstadt.add_node("95_V42",pos=[311.5,1624],signal=14,starter=True)
Darmstadt.add_node("95_E3",pos=[324,1607],ende=True)
Darmstadt.add_node("95_E4",pos=[311.5,1626],ende=True)
Darmstadt.add_edge("95_D12","96_D41",weight=1)
Darmstadt.add_edge("95_D12","96_D42",weight=1)
Darmstadt.add_edge("95_D20","136_D21",weight=1)
Darmstadt.add_edge("95_D22","95_E3",weight=1)
Darmstadt.add_edge("95_D11","95_E3",weight=1)
Darmstadt.add_edge("95_D11","95_E4",weight=1)
Darmstadt.add_edge("95_V42","96_D41",weight=1)
Darmstadt.add_edge("95_D21","95_E4",weight=1)

# 134
Darmstadt.add_node("134_E4",pos=[947,1428])
Darmstadt.add_node("134_D11",pos=[955.5,1435],signal=1,starter=True)
Darmstadt.add_node("134_D31",pos=[954,1411],signal=2,starter=True)
Darmstadt.add_node("134_E2",pos=[966,1423])
Darmstadt.add_node("134_A4",pos=[946,1426.5])
Darmstadt.add_node("134_A2",pos=[965,1424.5])
Darmstadt.add_edge("134_E4","10_D21",weight=1)
Darmstadt.add_edge("134_E4","10_D22",weight=1)
Darmstadt.add_edge("134_D11","134_E4",weight=1)
Darmstadt.add_edge("134_D11","134_E2",weight=1)
Darmstadt.add_edge("134_D31","134_E4",weight=1)
Darmstadt.add_edge("134_D31","134_E2",weight=1)
Darmstadt.add_edge("134_A2","134_E4",weight=1)
Darmstadt.add_edge("134_A4","134_E2",weight=1)
Darmstadt.add_edge("134_E2","9_D41",weight=1)

# 9

Darmstadt.add_node("9_E4",pos=[1097,1391])
Darmstadt.add_node("9_D41",pos=[1098,1389],signal=2)
Darmstadt.add_node("9_D21",pos=[1121,1387.5],signal=1,starter=True)
Darmstadt.add_node("9_E2",pos=[1120.5,1385.5],ende=True)
Darmstadt.add_node("9_D11",pos=[1108.5,1399.5])
Darmstadt.add_node("9_D31",pos=[1109.5,1377])
Darmstadt.add_node("9_E1",pos=[1110,1399])
Darmstadt.add_node("9_E3",pos=[1108,1376],ende=True)
Darmstadt.add_edge("9_D41","9_E2",weight=1)
Darmstadt.add_edge("9_D21","9_E4",weight=1)
Darmstadt.add_edge("9_D41","9_E1",weight=1)
Darmstadt.add_edge("9_D41","9_E3",weight=1)
Darmstadt.add_edge("9_D21","9_E4",weight=1)
Darmstadt.add_edge("9_D21","9_E1",weight=1)
Darmstadt.add_edge("9_D21","9_E3",weight=1)
Darmstadt.add_edge("9_D11","9_E4",weight=1)
Darmstadt.add_edge("9_D11","9_E2",weight=1)
Darmstadt.add_edge("9_D11","9_E3",weight=1)
Darmstadt.add_edge("9_D31","9_E1",weight=1)
Darmstadt.add_edge("9_D31","9_E2",weight=1)
Darmstadt.add_edge("9_D31","9_E4",weight=1)
Darmstadt.add_edge("9_E4","134_A2",weight=1)
Darmstadt.add_edge("9_E1","8_D31",weight=1)
Darmstadt.add_edge("9_E1","8_D32",weight=1)

# 144
Darmstadt.add_node("144_D11.1",pos=[1667.5,2335],signal=1,starter=True)
Darmstadt.add_node("144_D21",pos=[1694.5,2347],signal=2)
Darmstadt.add_node("144_D22",pos=[1684,2335],signal=4) 
Darmstadt.add_node("144_V30",pos=[1690.5,2320.5],signal=6)#1703.5,2326.5
Darmstadt.add_node("144_D31",pos=[1686.5,2324.5],signal=7)#1697.5,2330.5
Darmstadt.add_node("144_D32",pos=[1685,2323],signal=8)#1696,2329
Darmstadt.add_node("144_D41",pos=[1654,2312.5],signal=10)#1664,2317.5
Darmstadt.add_node("144_D42",pos=[1666.5,2326],signal=12)#1676.5,2331
Darmstadt.add_node("144_E4",pos=[1663,2326])
Darmstadt.add_node("144_E3",pos=[1685,2320.5])
Darmstadt.add_node("144_E2",pos=[1690,2335])
Darmstadt.add_node("144_E1",pos=[1669,2336],ende=True)
Darmstadt.add_edge("144_D11.1","144_E4",weight=1)
Darmstadt.add_edge("144_D11.1","144_E3",weight=1)
Darmstadt.add_edge("144_D11.1","144_E2",weight=1)
Darmstadt.add_edge("144_D21","144_E1",weight=1)
Darmstadt.add_edge("144_D21","144_E4",weight=1)
Darmstadt.add_edge("144_D22","144_E3",weight=1)
Darmstadt.add_edge("144_D31","144_E1",weight=1)
Darmstadt.add_edge("144_D32","144_E4",weight=1)
Darmstadt.add_edge("144_V30","144_E2",weight=1)
Darmstadt.add_edge("144_D41","144_E2",weight=1)
Darmstadt.add_edge("144_D41","144_E3",weight=1)
Darmstadt.add_edge("144_D42","144_E1",weight=1)
Darmstadt.add_edge("144_E4","142_D21",weight=1)
Darmstadt.add_edge("144_E4","142_D22",weight=1)
Darmstadt.add_edge("144_E4","142_D23",weight=1)
Darmstadt.add_edge("144_E4","142_D24",weight=1)
Darmstadt.add_edge("144_E2","146_D42",weight=1)
Darmstadt.add_edge("144_E3","90_D21",weight=1)
Darmstadt.add_edge("144_E3","90_D22",weight=1)

# 146
Darmstadt.add_node("146_D42",pos=[2373,2335],signal=12)
Darmstadt.add_node("146_D11",pos=[2391.5,2343.5],signal=1,starter=True)
Darmstadt.add_node("146_D12",pos=[2400.5,2352.5],signal=4,starter=True)
Darmstadt.add_node("146_D31",pos=[2376,2326.5],signal=7)
Darmstadt.add_node("146_D32",pos=[2374,2327.5],signal=9)
Darmstadt.add_node("146_E1",pos=[2375.5,2338])
Darmstadt.add_node("146_E2",pos=[2372,2328.5])
Darmstadt.add_node("146_E3",pos=[2395.5,2343.5],ende=True)
Darmstadt.add_edge("146_D11","146_E1",weight=1)
Darmstadt.add_edge("146_D12","146_E2",weight=1)
Darmstadt.add_edge("146_D31","146_E3",weight=1)
Darmstadt.add_edge("146_D32","146_E1",weight=1)
Darmstadt.add_edge("146_D42","146_E2",weight=1)
Darmstadt.add_edge("146_D42","146_E3",weight=1)
Darmstadt.add_edge("146_E1","144_D21",weight=1)
Darmstadt.add_edge("146_E1","144_D22",weight=1)
Darmstadt.add_edge("146_E2","70_D11",weight=1)
Darmstadt.add_edge("146_E2","70_D12",weight=1)
Darmstadt.add_edge("146_E2","70_D13",weight=1)

# 90
Darmstadt.add_node("90_D21",pos=[1476.5,1616.5],signal=1)
Darmstadt.add_node("90_D22",pos=[1479,1620],signal=2)
Darmstadt.add_node("90_V51",pos=[1496.5,1600],signal=5)
Darmstadt.add_node("90_D81",pos=[1471.5,1594.5],signal=6,starter=True)
Darmstadt.add_node("90_V112",pos=[1453,1609],signal=1)
Darmstadt.add_node("90_V111",pos=[1452.5,1608],signal=1)
Darmstadt.add_node("90_E1",pos=[1481.5,1616.5])
Darmstadt.add_node("90_E2",pos=[1496.5,1597.5])
Darmstadt.add_node("90_E3",pos=[1469,1594.5],ende=True)#1695
Darmstadt.add_node("90_E4",pos=[1453,1611])
Darmstadt.add_edge("90_E4","8_D21",weight=1)
Darmstadt.add_edge("90_E4","8_D22",weight=1)
Darmstadt.add_edge("90_E4","8_D51",weight=1)
Darmstadt.add_edge("90_E1","144_V30",weight=1)
Darmstadt.add_edge("90_E1","144_D31",weight=1)
Darmstadt.add_edge("90_E1","144_D32",weight=1)
Darmstadt.add_edge("90_V112","90_E1",weight=1)
Darmstadt.add_edge("90_V111","90_E2",weight=1)
Darmstadt.add_edge("90_V111","90_E3",weight=1)
Darmstadt.add_edge("90_D21","90_E4",weight=1)
Darmstadt.add_edge("90_D21","90_E3",weight=1)
Darmstadt.add_edge("90_D22","90_E2",weight=1)
Darmstadt.add_edge("90_V51","90_E1",weight=1)
Darmstadt.add_edge("90_V51","90_E4",weight=1)
Darmstadt.add_edge("90_D81","90_E1",weight=1)
Darmstadt.add_edge("90_D81","90_E2",weight=1)
Darmstadt.add_edge("90_D81","90_E4",weight=1)
Darmstadt.add_edge("90_E2","69_D116",weight=1)
Darmstadt.add_edge("90_E2","69_D115",weight=1)

# 69
Darmstadt.add_node("69_D116",pos=[1624,1572.5],signal=13)
Darmstadt.add_node("69_D115",pos=[1623,1569.5],signal=14)
Darmstadt.add_node("69_V51",pos=[1674.5,1557.5],signal=1)
Darmstadt.add_node("69_V52",pos=[1673,1556],signal=2)
Darmstadt.add_node("69_V84",pos=[1652.5,1537],signal=6,starter=True)
Darmstadt.add_node("69_V83",pos=[1654,1537],signal=7,starter=True)
Darmstadt.add_node("69_E2",pos=[1671.5,1554.5])
Darmstadt.add_node("69_E3",pos=[1650,1537],ende=True)
Darmstadt.add_node("69_E4",pos=[1625,1575])
Darmstadt.add_edge("69_E4","90_V51",weight=1)
Darmstadt.add_edge("69_D115","69_E3",weight=1)
Darmstadt.add_edge("69_D116","69_E2",weight=1)
Darmstadt.add_edge("69_V84","69_E4",weight=1)
Darmstadt.add_edge("69_V83","69_E2",weight=1)
Darmstadt.add_edge("69_V52","69_E3",weight=1)
Darmstadt.add_edge("69_V51","69_E4",weight=1)
Darmstadt.add_edge("69_E2","70_D42",weight=1)
Darmstadt.add_edge("69_E2","70_D41",weight=1)


# 70
Darmstadt.add_node("70_D11",pos=[1960,1468.5],signal=1)
Darmstadt.add_node("70_D12",pos=[1962.5,1469.5],signal=2)
Darmstadt.add_node("70_D13",pos=[1964,1469],signal=3)
Darmstadt.add_node("70_D21",pos=[1982,1443.5],signal=4)
Darmstadt.add_node("70_D22",pos=[1971,1447.5],signal=5)
Darmstadt.add_node("70_D31",pos=[1957,1441],signal=6,starter=True)
Darmstadt.add_node("70_D32",pos=[1955.5,1441.5],signal=7,starter=True)
Darmstadt.add_node("70_D41",pos=[1937,1460],signal=8)
Darmstadt.add_node("70_D42",pos=[1949,1457],signal=9)
Darmstadt.add_node("70_E1",pos=[1967,1470])
Darmstadt.add_node("70_E2",pos=[1971,1445],ende=True)
Darmstadt.add_node("70_E3",pos=[1954,1442],ende=True)
Darmstadt.add_node("70_E4",pos=[1937,1464])
Darmstadt.add_edge("70_E4","69_V51",weight=1)
Darmstadt.add_edge("70_E4","69_V52",weight=1)
Darmstadt.add_edge("70_E1","146_D31",weight=1)
Darmstadt.add_edge("70_E1","146_D32",weight=1)
Darmstadt.add_edge("70_D11","70_E4",weight=1)
Darmstadt.add_edge("70_D12","70_E3",weight=1)
Darmstadt.add_edge("70_D13","70_E2",weight=1)
Darmstadt.add_edge("70_D21","70_E1",weight=1)
Darmstadt.add_edge("70_D21","70_E4",weight=1)
Darmstadt.add_edge("70_D22","70_E3",weight=1)
Darmstadt.add_edge("70_D31","70_E2",weight=1)
Darmstadt.add_edge("70_D32","70_E1",weight=1)
Darmstadt.add_edge("70_D41","70_E2",weight=1)
Darmstadt.add_edge("70_D41","70_E3",weight=1)
Darmstadt.add_edge("70_D42","70_E1",weight=1)
Darmstadt.add_edge("70_E2","37_D21",weight=1)
Darmstadt.add_edge("70_E2","37_D22.1",weight=1)


#nx.write_edgelist(Darmstadt,".\\TestDarmstadt.edgelist",data=['weight','signal'])
#nx.write_graphml_lxml(Darmstadt, ".\\TestDarmstadt.graphml", prettyprint=True)
#nx.write_gml(Darmstadt,".\\TestDarmstadt.gml")


# 37
Darmstadt.add_node("37_D111",pos=[2122.5,1311],signal=16)
Darmstadt.add_node("37_D112",pos=[2121,1312.5],signal=17)
Darmstadt.add_node("37_D21",pos=[2126.5,1325.5],signal=9)
Darmstadt.add_node("37_D22.1",pos=[2128.5,1326.5],signal=10)
Darmstadt.add_node("37_D51",pos=[2146.5,1325.5],signal=20,starter=True)
Darmstadt.add_node("37_D52",pos=[2146.5,1323.5],signal=21,starter=True)
Darmstadt.add_node("37_D81",pos=[2137.5,1307],signal=13,starter=True)
Darmstadt.add_node("37_D82.1",pos=[2136,1306],signal=14,starter=True)
Darmstadt.add_node("37_E1",pos=[2128.5,1326.5])
Darmstadt.add_node("37_E2",pos=[2146.5,1321.5],ende=True)
Darmstadt.add_node("37_E3",pos=[2134.5,1306],ende=True)
Darmstadt.add_node("37_E4",pos=[2120,1314],ende=True)
Darmstadt.add_edge("37_D112","37_E1",weight=1)
Darmstadt.add_edge("37_D111","37_E2",weight=1)
Darmstadt.add_edge("37_D111","37_E3",weight=1)
Darmstadt.add_edge("37_D82.1","37_E4",weight=1)
Darmstadt.add_edge("37_D81","37_E1",weight=1)
Darmstadt.add_edge("37_D81","37_E2",weight=1)
Darmstadt.add_edge("37_D52","37_E3",weight=1)
Darmstadt.add_edge("37_D51","37_E1",weight=1)
Darmstadt.add_edge("37_D51","37_E4",weight=1)
Darmstadt.add_edge("37_D21","37_E3",weight=1)
Darmstadt.add_edge("37_D21","37_E4",weight=1)
Darmstadt.add_edge("37_D22.1","37_E2",weight=1)
Darmstadt.add_edge("37_E1","70_D21",weight=1)
Darmstadt.add_edge("37_E1","70_D21",weight=1)
"""

# 61
Darmstadt.add_node("61_D23",pos=[],signal=3)
Darmstadt.add_node("61_D82",pos=[],signal=6)
Darmstadt.add_node("61_D83",pos=[],signal=7)

# 150
Darmstadt.add_node("150_V21",pos=[],signal=1)
Darmstadt.add_node("150_V81",pos=[],signal=2)

# 111
Darmstadt.add_node("111_D11",pos=[],signal=1)
Darmstadt.add_node("111_D21",pos=[],signal=2)
Darmstadt.add_node("111_D31",pos=[],signal=3)
Darmstadt.add_node("111_D41",pos=[],signal=4)
Darmstadt.add_node("111_E1",pos=[])
Darmstadt.add_node("111_E2",pos=[])
Darmstadt.add_node("111_E3",pos=[])

# 173
Darmstadt.add_node("173_V11",pos=[],signal=2)
Darmstadt.add_node("173_V12",pos=[],signal=3)
Darmstadt.add_node("173_V5",pos=[],signal=1)

# 33
Darmstadt.add_node("33_D11",pos=[],signal=1)
Darmstadt.add_node("33_D12",pos=[],signal=3)
Darmstadt.add_node("33_D21",pos=[],signal=5)
Darmstadt.add_node("33_D22",pos=[],signal=9)
Darmstadt.add_node("33_D31",pos=[],signal=11)
Darmstadt.add_node("33_D32",pos=[],signal=14)
Darmstadt.add_node("33_D41",pos=[],signal=16)
Darmstadt.add_node("33_D42",pos=[],signal=19)

# 126 TODO Anfragen
Darmstadt.add_node("126_D11",pos=[],signal=1)
Darmstadt.add_node("126_D12",pos=[],signal=2)
Darmstadt.add_node("126_D21",pos=[],signal=3)
Darmstadt.add_node("126_D31",pos=[],signal=4)
Darmstadt.add_node("126_D32",pos=[],signal=5)
Darmstadt.add_node("126_D41",pos=[],signal=6)
Darmstadt.add_node("126_E2",pos=[])
Darmstadt.add_node("126_E3",pos=[])
Darmstadt.add_node("126_E4",pos=[])


# 76 TODO Anfragen
Darmstadt.add_node("76_D11.1",pos=[],signal=6)
Darmstadt.add_node("76_D11.2",pos=[],signal=7)
Darmstadt.add_node("76_D5.1",pos=[],signal=3)
Darmstadt.add_node("76_D5.2",pos=[],signal=4)
Darmstadt.add_node("76_D2.1",pos=[],signal=1)
Darmstadt.add_node("76_D2.1",pos=[],signal=2)
Darmstadt.add_node("76_D8",pos=[],signal=5)
Darmstadt.add_node("76_E1",pos=[])
Darmstadt.add_node("76_E2",pos=[])
Darmstadt.add_node("76_E3",pos=[])
Darmstadt.add_node("76_E4",pos=[])

# 11
Darmstadt.add_node("11_V21",pos=[],signal=1)
Darmstadt.add_node("11_V22",pos=[],signal=2)
Darmstadt.add_node("11_D51",pos=[],signal=3)
Darmstadt.add_node("11_D81",pos=[],signal=4)
Darmstadt.add_node("11_D82",pos=[],signal=5)
Darmstadt.add_node("11_D91",pos=[],signal=6)
Darmstadt.add_node("11_D41",pos=[],signal=7)
Darmstadt.add_node("11_D42.1",pos=[],signal=8)
Darmstadt.add_node("11_E1",pos=[])
Darmstadt.add_node("11_E2",pos=[])
Darmstadt.add_node("11_E3",pos=[])
Darmstadt.add_node("11_E4",pos=[])

# 110
Darmstadt.add_node("110_D11",pos=[],signal=1)
Darmstadt.add_node("110_D21",pos=[],signal=2)
Darmstadt.add_node("110_D21",pos=[],signal=3)
Darmstadt.add_node("110_E1",pos=[])
Darmstadt.add_node("110_E3",pos=[])
Darmstadt.add_node("110_E4",pos=[])


# 161
Darmstadt.add_node("161_DK41",pos=[],signal=1)
Darmstadt.add_node("161_DK61",pos=[],signal=3)
Darmstadt.add_node("161_DK71",pos=[],signal=5)
Darmstadt.add_node("161_DK91",pos=[],signal=7)
Darmstadt.add_node("161_DK101",pos=[],signal=9)
Darmstadt.add_node("161_DK121",pos=[],signal=11)
Darmstadt.add_node("161_E2",pos=[])
Darmstadt.add_node("161_E3",pos=[])
Darmstadt.add_node("161_E4",pos=[])

# 160
Darmstadt.add_node("160_IR21",pos=[],signal=1)
Darmstadt.add_node("160_D31",pos=[],signal=2)
Darmstadt.add_node("160_D41",pos=[],signal=4)
Darmstadt.add_node("160_E2",pos=[])
Darmstadt.add_node("160_E3",pos=[])
Darmstadt.add_node("160_E4",pos=[])

# 75
Darmstadt.add_node("75_D21",pos=[],signal=3)
Darmstadt.add_node("75_D22",pos=[],signal=4)
Darmstadt.add_node("75_D51",pos=[],signal=11)
Darmstadt.add_node("75_V101",pos=[],signal=13)
Darmstadt.add_node("75_D111",pos=[],signal=14)
Darmstadt.add_node("75_D121_1",pos=[],signal=16)
Darmstadt.add_node("75_E1",pos=[])
Darmstadt.add_node("75_E2",pos=[])
Darmstadt.add_node("75_E4",pos=[])

# 174
Darmstadt.add_node("174_D51",pos=[],signal=1)
Darmstadt.add_node("174_D101",pos=[],signal=3)
Darmstadt.add_node("174_E2",pos=[])
Darmstadt.add_node("174_E4",pos=[])

# 46
Darmstadt.add_node("46_D11",pos=[],signal=1)
Darmstadt.add_node("46_D12",pos=[],signal=2)
Darmstadt.add_node("46_D51",pos=[],signal=8)
Darmstadt.add_node("46_D81",pos=[],signal=10)
Darmstadt.add_node("46_D82",pos=[],signal=11)
Darmstadt.add_node("46_D111",pos=[],signal=15)
Darmstadt.add_node("46_D121",pos=[],signal=17)
Darmstadt.add_node("46_E1",pos=[])
Darmstadt.add_node("46_E2",pos=[])
Darmstadt.add_node("46_E3",pos=[])
Darmstadt.add_node("46_E4",pos=[])

# 24
Darmstadt.add_node("24_D21",pos=[],signal=1)
Darmstadt.add_node("24_D22",pos=[],signal=2)
Darmstadt.add_node("24_D81",pos=[],signal=5)
Darmstadt.add_node("24_D82",pos=[],signal=6)
Darmstadt.add_node("24_D111",pos=[],signal=9)
Darmstadt.add_node("24_D112",pos=[],signal=10)
Darmstadt.add_node("24_E1",pos=[])
Darmstadt.add_node("24_E3",pos=[])
Darmstadt.add_node("24_E4",pos=[])

# 12
Darmstadt.add_node("12_D11",pos=[],signal=1)
Darmstadt.add_node("12_D12",pos=[],signal=2)
Darmstadt.add_node("12_D13",pos=[],signal=3)
Darmstadt.add_node("12_D21",pos=[],signal=4)
Darmstadt.add_node("12_D22",pos=[],signal=5)
Darmstadt.add_node("12_D31",pos=[],signal=6)
Darmstadt.add_node("12_D32",pos=[],signal=7)
Darmstadt.add_node("12_D33",pos=[],signal=8)
Darmstadt.add_node("12_D42",pos=[],signal=10)
Darmstadt.add_node("12_D28",pos=[],signal=11)
Darmstadt.add_node("12_D29",pos=[],signal=12)
Darmstadt.add_node("12_E1",pos=[])
Darmstadt.add_node("12_E2",pos=[])
Darmstadt.add_node("12_E4",pos=[])


# 59
Darmstadt.add_node("59_D21",pos=[],signal=1)
Darmstadt.add_node("59_D51",pos=[],signal=2)
Darmstadt.add_node("59_D81",pos=[],signal=3)
Darmstadt.add_node("59_D111",pos=[],signal=4)
Darmstadt.add_node("59_E1",pos=[])
Darmstadt.add_node("59_E2",pos=[])
Darmstadt.add_node("59_E3",pos=[])
Darmstadt.add_node("59_E4",pos=[])

# 162
Darmstadt.add_node("162_V22",pos=[],signal=2)
Darmstadt.add_node("162_V82",pos=[],signal=4)
Darmstadt.add_node("162_D91_1",pos=[],signal=5)
Darmstadt.add_node("162_D111",pos=[],signal=8)
Darmstadt.add_node("162_D112",pos=[],signal=9)
Darmstadt.add_node("162_E1",pos=[])
Darmstadt.add_node("162_E3",pos=[])
Darmstadt.add_node("162_E4",pos=[])

# 163
Darmstadt.add_node("163_D21",pos=[],signal=1)
Darmstadt.add_node("163_D22",pos=[],signal=2)
Darmstadt.add_node("163_D41",pos=[],signal=3)
Darmstadt.add_node("163_D51",pos=[],signal=4)
Darmstadt.add_node("163_D81",pos=[],signal=5)
Darmstadt.add_node("163_E1",pos=[])
Darmstadt.add_node("163_E2",pos=[])
Darmstadt.add_node("163_E3",pos=[])

# 104
Darmstadt.add_node("104_D1",pos=[],signal=1)
Darmstadt.add_node("104_D2",pos=[],signal=2)
Darmstadt.add_node("104_D3",pos=[],signal=3)
Darmstadt.add_node("104_D4",pos=[],signal=4)
Darmstadt.add_node("104_D5",pos=[],signal=5)
Darmstadt.add_node("104_D6",pos=[],signal=6)
Darmstadt.add_node("104_V211",pos=[],signal=15)
Darmstadt.add_node("104_V212",pos=[],signal=16)
Darmstadt.add_node("104_V811",pos=[],signal=17)
Darmstadt.add_node("104_V812",pos=[],signal=18)
Darmstadt.add_node("104_V1111",pos=[],signal=19)
Darmstadt.add_node("104_V1112",pos=[],signal=20)
Darmstadt.add_node("104_E1",pos=[])
Darmstadt.add_node("104_E3",pos=[])
Darmstadt.add_node("104_E4",pos=[])


# 23

Darmstadt.add_node("23_V21",pos=[],signal=1)
Darmstadt.add_node("23_V22",pos=[],signal=2)
Darmstadt.add_node("23_D54",pos=[],signal=10)
Darmstadt.add_node("23_D55",pos=[],signal=11)
Darmstadt.add_node("23_D63",pos=[],signal=12)
Darmstadt.add_node("23_D64",pos=[],signal=13)
Darmstadt.add_node("23_V81",pos=[],signal=7)
Darmstadt.add_node("23_V82",pos=[],signal=8)
Darmstadt.add_node("23_E1",pos=[])
Darmstadt.add_node("23_E3",pos=[])
Darmstadt.add_node("23_E4",pos=[])


# 28
Darmstadt.add_node("28_V12",pos=[],signal=2)
Darmstadt.add_node("28_V21",pos=[],signal=3)
Darmstadt.add_node("28_V22",pos=[],signal=4)
Darmstadt.add_node("28_V23",pos=[],signal=5)
Darmstadt.add_node("28_V24",pos=[],signal=6)
Darmstadt.add_node("28_V31",pos=[],signal=7)
Darmstadt.add_node("28_V32",pos=[],signal=8)
Darmstadt.add_node("28_E1",pos=[])
Darmstadt.add_node("28_E3",pos=[])
Darmstadt.add_node("28_E4",pos=[])

# 29
Darmstadt.add_node("29_D11",pos=[],signal=1)
Darmstadt.add_node("29_D12",pos=[],signal=2)
Darmstadt.add_node("29_E3",pos=[])
Darmstadt.add_node("29_E4",pos=[])

# 30
Darmstadt.add_node("30_V11",pos=[],signal=17)
Darmstadt.add_node("30_V11",pos=[],signal=18)
Darmstadt.add_node("30_V11",pos=[],signal=19)
Darmstadt.add_node("30_E1",pos=[])
Darmstadt.add_node("30_E2",pos=[])

# 32
Darmstadt.add_node("32_D51",pos=[],signal=1)
Darmstadt.add_node("32_D52",pos=[],signal=2)
Darmstadt.add_node("32_D71",pos=[],signal=5)
Darmstadt.add_node("32_D91",pos=[],signal=7)
Darmstadt.add_node("32_D92",pos=[],signal=8)
Darmstadt.add_node("32_E4",pos=[])

# 34
Darmstadt.add_node("34_D41",pos=[],signal=5)
Darmstadt.add_node("34_D42",pos=[],signal=6)
Darmstadt.add_node("34_D81",pos=[],signal=7)
Darmstadt.add_node("34_D82",pos=[],signal=8)
Darmstadt.add_node("34_D01",pos=[],signal=10)
Darmstadt.add_node("34_E1",pos=[])

# 36
Darmstadt.add_node("36_D11",pos=[],signal=1)
Darmstadt.add_node("36_D12",pos=[],signal=2)
Darmstadt.add_node("36_D13",pos=[],signal=3)
Darmstadt.add_node("36_D14",pos=[],signal=4)
Darmstadt.add_node("36_D21",pos=[],signal=8)
Darmstadt.add_node("36_D22",pos=[],signal=9)
Darmstadt.add_node("36_D23",pos=[],signal=10)
Darmstadt.add_node("36_D24",pos=[],signal=11)
Darmstadt.add_node("36_D31",pos=[],signal=16)
Darmstadt.add_node("36_D32",pos=[],signal=17)
Darmstadt.add_node("36_D33",pos=[],signal=18)
Darmstadt.add_node("36_D34",pos=[],signal=19)
Darmstadt.add_node("36_D40",pos=[],signal=24)
Darmstadt.add_node("36_D41",pos=[],signal=25)
Darmstadt.add_node("36_D42",pos=[],signal=26)
Darmstadt.add_node("36_D43",pos=[],signal=27)
Darmstadt.add_node("36_E1",pos=[])
Darmstadt.add_node("36_E2",pos=[])
Darmstadt.add_node("36_E3",pos=[])
Darmstadt.add_node("36_E4",pos=[])

# 35
Darmstadt.add_node("35_D11",pos=[],signal=1)
Darmstadt.add_node("35_D21",pos=[],signal=2)
Darmstadt.add_node("35_D31",pos=[],signal=3)
Darmstadt.add_node("35_D41",pos=[],signal=4)
Darmstadt.add_node("35_E1",pos=[])
Darmstadt.add_node("35_E2",pos=[])
Darmstadt.add_node("35_E3",pos=[])
Darmstadt.add_node("35_E4",pos=[])

# 141
Darmstadt.add_node("141_D11.1",pos=[],signal=1)
Darmstadt.add_node("141_V51",pos=[],signal=5)
Darmstadt.add_node("141_V52",pos=[],signal=6)
Darmstadt.add_node("141_V111",pos=[],signal=7)
Darmstadt.add_node("141_V121",pos=[],signal=9)
Darmstadt.add_node("141_E1",pos=[])
Darmstadt.add_node("141_E2",pos=[])
Darmstadt.add_node("141_E4",pos=[])

# 131
Darmstadt.add_node("131_D1",pos=[],signal=1)
Darmstadt.add_node("131_D2",pos=[],signal=2)
Darmstadt.add_node("131_E2",pos=[])
Darmstadt.add_node("131_E4",pos=[])

# 1
Darmstadt.add_node("1_D11",pos=[],signal=1)
Darmstadt.add_node("1_D12",pos=[],signal=2)
Darmstadt.add_node("1_D21",pos=[],signal=3)
Darmstadt.add_node("1_D22",pos=[],signal=4)
Darmstadt.add_node("1_D23",pos=[],signal=5)
Darmstadt.add_node("1_D33",pos=[],signal=6)
Darmstadt.add_node("1_D34",pos=[],signal=7)
Darmstadt.add_node("1_D35",pos=[],signal=8)
Darmstadt.add_node("1_D41",pos=[],signal=9)
Darmstadt.add_node("1_D42",pos=[],signal=10)
Darmstadt.add_node("1_D43",pos=[],signal=11)
Darmstadt.add_node("1_E1",pos=[])
Darmstadt.add_node("1_E2",pos=[])
Darmstadt.add_node("1_E3",pos=[])
Darmstadt.add_node("1_E4",pos=[])

# 2
Darmstadt.add_node("2_D11",pos=[],signal=1)
Darmstadt.add_node("2_D12",pos=[],signal=2)
Darmstadt.add_node("2_D13",pos=[],signal=3)
Darmstadt.add_node("2_V21",pos=[],signal=4)
Darmstadt.add_node("2_V22",pos=[],signal=5)
Darmstadt.add_node("2_V23",pos=[],signal=6)
Darmstadt.add_node("2_V24",pos=[],signal=7)
Darmstadt.add_node("2_V31",pos=[],signal=8)
Darmstadt.add_node("2_V32",pos=[],signal=9)
Darmstadt.add_node("2_V33",pos=[],signal=10)
Darmstadt.add_node("2_V41",pos=[],signal=11)
Darmstadt.add_node("2_V42",pos=[],signal=12)
Darmstadt.add_node("2_V43",pos=[],signal=13)
Darmstadt.add_node("2_E1",pos=[])
Darmstadt.add_node("2_E2",pos=[])
Darmstadt.add_node("2_E3",pos=[])
Darmstadt.add_node("2_E4",pos=[])

# 7
Darmstadt.add_node("7_D21",pos=[],signal=1)
Darmstadt.add_node("7_D22",pos=[],signal=2)
Darmstadt.add_node("7_D41",pos=[],signal=3)
Darmstadt.add_node("7_D42",pos=[],signal=4)
Darmstadt.add_node("7_E2",pos=[])
Darmstadt.add_node("7_E4",pos=[])

# 3 
Darmstadt.add_node("3_V10",pos=[],signal=30)
Darmstadt.add_node("3_D11",pos=[],signal=1)
Darmstadt.add_node("3_D12",pos=[],signal=2)
Darmstadt.add_node("3_D13",pos=[],signal=3)
Darmstadt.add_node("3_D21",pos=[],signal=4)
Darmstadt.add_node("3_D22",pos=[],signal=5)
Darmstadt.add_node("3_D23",pos=[],signal=6)
Darmstadt.add_node("3_D31",pos=[],signal=7)
Darmstadt.add_node("3_D32",pos=[],signal=8)
Darmstadt.add_node("3_D33",pos=[],signal=9)
Darmstadt.add_node("3_V41",pos=[],signal=10)
Darmstadt.add_node("3_V42",pos=[],signal=11)
Darmstadt.add_node("3_D43",pos=[],signal=12)
Darmstadt.add_node("3_E1",pos=[])
Darmstadt.add_node("3_E2",pos=[])
Darmstadt.add_node("3_E3",pos=[])
Darmstadt.add_node("3_E4",pos=[])

# 4
Darmstadt.add_node("4_D21",pos=[],signal=2)
Darmstadt.add_node("4_D22",pos=[],signal=3)
Darmstadt.add_node("4_D31",pos=[],signal=8)
Darmstadt.add_node("4_D51",pos=[],signal=10)
Darmstadt.add_node("4_D81",pos=[],signal=13)
Darmstadt.add_node("4_D82",pos=[],signal=14)
Darmstadt.add_node("4_D83",pos=[],signal=15)
Darmstadt.add_node("4_D91",pos=[],signal=21)
Darmstadt.add_node("4_D92",pos=[],signal=22)
Darmstadt.add_node("4_D111",pos=[],signal=25)
Darmstadt.add_node("4_D113",pos=[],signal=27)
Darmstadt.add_node("4_D121",pos=[],signal=33)
Darmstadt.add_node("4_D122",pos=[],signal=34)
Darmstadt.add_node("4_E1",pos=[])
Darmstadt.add_node("4_E2",pos=[])
Darmstadt.add_node("4_E3",pos=[])
Darmstadt.add_node("4_E4",pos=[])

# 5
Darmstadt.add_node("5_D11",pos=[],signal=1)
Darmstadt.add_node("5_D12",pos=[],signal=2)
Darmstadt.add_node("5_D21",pos=[],signal=3)
Darmstadt.add_node("5_D31",pos=[],signal=4)
Darmstadt.add_node("5_D41",pos=[],signal=5)
Darmstadt.add_node("5_D42",pos=[],signal=6)
Darmstadt.add_node("5_D43",pos=[],signal=12)
Darmstadt.add_node("5_E1",pos=[])
Darmstadt.add_node("5_E2",pos=[])
Darmstadt.add_node("5_E3",pos=[])
Darmstadt.add_node("5_E4",pos=[])

# 6
Darmstadt.add_node("6_D1",pos=[],signal=1)
Darmstadt.add_node("6_D2",pos=[],signal=2)
Darmstadt.add_node("6_D6",pos=[],signal=6)
Darmstadt.add_node("6_D8",pos=[],signal=8)
Darmstadt.add_node("6_D10",pos=[],signal=10)
Darmstadt.add_node("6_D12",pos=[],signal=12)
Darmstadt.add_node("6_D22",pos=[],signal=22)
Darmstadt.add_node("6_D24",pos=[],signal=24)
Darmstadt.add_node("6_D26",pos=[],signal=26)

# 81
#Darmstadt.add_node("81_D11")
"""

import re
import pygsp as ps 
n_eigenvectors = 7


pos = nx.get_node_attributes(Darmstadt,'pos')
sigs = nx.get_node_attributes(Darmstadt,'signal')
A = nx.adjacency_matrix(Darmstadt)
fig,ax = plt.subplots(1,1) 
#ax[0].spy(A)
nx.draw(Darmstadt,pos=pos,node_size=10,ax=ax)
#nx.draw_networkx_labels(Darmstadt,pos,font_size=10,ax=ax)
nx.draw_networkx_nodes(Darmstadt,pos=pos,node_color='#a142f5',node_size=10,ax=ax)
#nx.draw_networkx_labels(Darmstadt,pos=pos,font_size=14,ax=ax)
nx.draw_networkx_edges(Darmstadt,pos=pos,edge_color="gray",alpha=0.5,ax=ax)
#nx.draw_networkx_edge_labels(G,pos=pos,edge_labels=labels,ax=ax)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout()
plt.show()

fig2,ax3 = plt.subplots(1,2,figsize=(18,14))

D_g = ps.graphs.Graph(A) 
#xs,ys,_ = darmi.get_ids()
D_g.set_coordinates([v for v in pos.values()])#zip(xs,ys)])
D_g.compute_differential_operator()
D_g.compute_fourier_basis()
D_g.compute_laplacian()

_ = D_g.plot(D_g.U[:,1],ax=ax3[0],vertex_size=30)
_ = ax3[0].set_title('EV 1')
ax3[0].set_axis_off()
_ = D_g.plot(D_g.U[:,2],ax=ax3[1],vertex_size=30)
_ = ax3[1].set_title('EV 2')
ax3[1].set_axis_off()
#D_g.plot(D_g.U[:,2],ax=ax3[2])
#D_g.plot(D_g.U[:,4],ax=ax3[3])
#D_g.plot(D_g.U[:,5],ax=ax3[4])
fig2.tight_layout()

#fig2.subplots_adjust(hspace=0.5,right=0.8)
#cax = fig.add_axes([0.82,0.16,0.01,0.26])
plt.show()
D1 = D_g.D.toarray() 
L = D_g.L.toarray()
for k in range(10):
    plt.plot(D_g.U[:,k])
plt.show()
plt.imshow(D_g.W.toarray())
plt.show()

import scipy.stats as ss

def gen_signal(Graph, sigma=0.1, kind="ball", size="big"):
    if kind == "ball" and size == "big":
        #sources = [2, 6, 74, 41, 20, 32, 31, 9, 10, 56, 17, 16, 18]
        sources = np.random.randint(0,220,(int(Graph.n_vertices*0.69)))
    elif kind == "line" and size == "big":
        sources = [20, 41, 74, 6, 16, 45, 68, 57, 15, 30, 11, 23, 43, 24]#,
                   #48, 12, 21, 13, 69, 37, 7, 3, 63]
    elif kind == "idk":
        sources = [19, 42, 45, 46, 66, 68, 70, 57, 15, 30, 65, 71]
    signal = np.zeros(Graph.n_vertices)
    signal[sources] = np.random.randint(0, 20, (np.shape(sources)))#1
    np.random.seed(43)
    noise = np.random.normal(0, sigma, Graph.n_vertices)
    #noise[noise <= 0] = 0
    #noise = np.arange(-15, 16)
    noiseU, noiseL = noise + 0.5, noise - 0.5
    prob = ss.norm.cdf(noiseU, scale=9) - ss.norm.cdf(noiseL, scale=9)
    prob = prob/prob.sum()
    noisy = signal + np.random.choice(noise, size=Graph.n_vertices, p=prob) 
    #noisy = signal + np.random.normal(0, sigma, Graph.n_vertices)
    noisy[noisy <= 0] = 0
    return noisy, sources, signal


nosilent = nx.get_node_attributes(Darmstadt,'signal')
sigIndx = [k for k,i in enumerate(Darmstadt.nodes().keys()) if i in nosilent]
silent = [k for k in range(D_g.n_vertices) if k not in sigIndx]
blind_spots = np.zeros(D_g.n_vertices)
blind_spots[silent] = 1
mask = np.ones(D_g.n_vertices)
mask[silent] = 0

ende = nx.get_node_attributes(Darmstadt,'ende')
sigIEnde = [k for k,i in enumerate(Darmstadt.nodes().keys()) if i  in ende]

starter = nx.get_node_attributes(Darmstadt,'starter')
sigIStart = [k for k,i in enumerate(Darmstadt.nodes().keys()) if i in starter]



def makeSignal(day, signalDictionary, signalList, Graph):
    y = np.zeros((Graph.n_vertices,1440))
    data = np.load('./Darmstadt_verkehr/SensorData_Sensor_Big_{}_Counts.npz'.format(day),allow_pickle=True)['arr_0'].reshape((1,))[0]
    for k, i in enumerate(signalDictionary):
        strID = re.split('_',i)[0]
        if len(strID) < 2:
            strID = 'A00' + strID
        elif len(strID) < 3:
            strID = 'A0' + strID
        else: 
            strID = 'A' + strID
        y[signalList[k],:] = data[strID]['signals'][:,signalDictionary[i]-1]
    return y 


import cvxpy as cp 
x = cp.Variable(D_g.n_edges)
s = cp.Variable(D_g.n_edges)

import pyunlocbox 
gamma = 3 
import tqdm 

datasigns = makeSignal('16_1_2020', nosilent, sigIndx, D_g)
DD = nx.incidence_matrix(Darmstadt,oriented=True)
Dm = np.maximum(DD.todense(),np.zeros((D_g.n_vertices,D_g.n_edges)))
Dp = -np.minimum(DD.todense(),np.zeros((D_g.n_vertices,D_g.n_edges)))
Dx = np.row_stack((Dm,Dp,Dp,-Dp,-Dm))
Ds = np.row_stack((np.zeros((3*D_g.n_vertices,D_g.n_edges)),Dp,-Dm))

def parseDay(Graph,Dx,Ds,measurements,sigIEnde,sigIStart):
    x = cp.Variable(Graph.n_edges)
    s = cp.Variable(Graph.n_edges)
    XX = np.zeros((Graph.n_edges,1440))
    SS = np.zeros((Graph.n_edges,1440))
    for k in tqdm.trange(1439):
        measure_k = measurements[:,k] #/n_max
        measure_k1 = measurements[:,k+1]#/n_max
        if np.isnan(measure_k1).any() or np.isnan(measure_k).any():
            XX[:,k] = np.zeros(Graph.n_edges)
            SS[:,k] = np.zeros(Graph.n_edges)
        #    return
            continue
        else:
            ym0 = np.copy(measure_k)
            ys0 = np.copy(measure_k1) 
            yd = measure_k1 - measure_k

            ym0[sigIEnde] = 0
            ys0[sigIStart] = 0
            y_bar = np.hstack((measure_k1,measure_k,yd,-ym0,-ys0))
            problem = cp.Minimize(cp.norm1(Dx*x + Ds*s - y_bar) + 0.001*cp.norm1(x) + 0.005*cp.norm1(s))
            prob = cp.Problem(problem)
            prob.solve(solver=cp.GUROBI,verbose=False,warmstart=True)
            XX[:,k] = x.value.copy()
            SS[:,k] = s.value.copy()
    return XX, SS

#XX, SS = parseDay(D_g,Dx,Ds,datasigns,sigIEnde,sigIStart)
XX = np.load("XX.npy")
SS = np.load("SS.npy")

print("Geladen")

""" Fertig
"""
# mse = np.zeros((3,11))
# mae = np.zeros((3,11))
# mse_t1, mse_t2, mse_t3 = 0,0,0
# mae_t1, mae_t2, mae_t3 = 0,0,0
# noise_level = 1/(10**(np.arange(-10,45,5)/20))
# counter = 0
# for sigm in range(11):
#     sigma = noise_level[sigm]
#     print("Sigma level {}".format(sigma))
#     for t in tqdm.trange(750,850):
#         tim = t
#         #n_max=np.amax(datasigns[:,tim:tim+1])
#         measure_k = datasigns[:,tim] #/n_max
#         measure_k1 = datasigns[:,tim+1]#/n_max
#         if np.isnan(measure_k1).any():
#             break
#         ym0 = np.copy(measure_k)
#         ys0 = np.copy(measure_k1) 
#         yd = measure_k1 - measure_k

#         ym0[sigIEnde] = 0
#         ys0[sigIStart] = 0
#         y_bar = np.hstack((measure_k1,measure_k,yd,-ym0,-ys0))
#         problem = cp.Minimize(cp.norm1(Dx*x + Ds*s - y_bar) + 0.001*cp.norm1(x) + 0.005*cp.norm1(s))
#         prob = cp.Problem(problem)
#         prob.solve(solver=cp.GUROBI,verbose=False,warmstart=True)
#         original_signal = np.squeeze(np.asarray(np.dot(Dp,x.value-s.value))) + np.squeeze(np.asarray(np.dot(-Dm,s.value)))*blind_spots

#         #np.random.seed(43)
#         noise = np.random.normal(0, sigma, D_g.n_vertices)
#         #noiseU, noiseL = noise + 0.5, noise - 0.5
#         #prob = ss.norm.cdf(noiseU, scale=9) - ss.norm.cdf(noiseL, scale=9)
#         #prob = prob/prob.sum()
#         noisy = original_signal + noise#np.random.choice(noise, size=D_g.n_vertices, p=prob) 
            

#         y1 = np.copy(measure_k) + np.random.normal(0, sigma, D_g.n_vertices)
#         y2 = np.copy(measure_k1) + np.random.normal(0, sigma, D_g.n_vertices)
#         y1[y1<0] = 0
#         y2[y2<0] = 0
#         yd = y2 - y1
#         ym = np.copy(y1)
#         ys = np.copy(y2)
#         ym[sigIEnde] = 0
#         ys[sigIStart] = 0
#         x1 = cp.Variable(D_g.n_edges)
#         s1 = cp.Variable(D_g.n_edges)

#         y_bar = np.hstack((y2,y1,yd,-ym,-ys))
#         problem2 = cp.Minimize(cp.norm1(Dx*x1 + Ds*s1 - y_bar) + 0.001*cp.norm1(x1) + 0.005*cp.norm1(s1))
#         prob2 = cp.Problem(problem2)
#         prob2.solve(solver=cp.GUROBI,verbose=False,warmstart=True)

#         original_signal2 = np.squeeze(np.asarray(np.dot(Dp,x1.value-s1.value))) + np.squeeze(np.asarray(np.dot(-Dm,s1.value)))*blind_spots

#         dd = pyunlocbox.functions.dummy()
#         rr = pyunlocbox.functions.norm_l1()
#         ff = pyunlocbox.functions.norm_l2(w=mask,y=y1, lambda_=gamma)
#         LL = D_g.D.T.toarray()
#         step = 0.999 / (1 + np.linalg.norm(LL))
#         solver= pyunlocbox.solvers.mlfbf(L=LL,step=step)
#         x0 = y1.copy() 
#         prob1 = pyunlocbox.solvers.solve([dd,rr,ff],solver=solver,x0=x0,rtol=0,maxit=1000,verbosity='NONE')

#         rr = pyunlocbox.functions.norm_l2(A=LL, tight=False)
#         step = 0.999 / np.linalg.norm(np.dot(LL.T, LL) + gamma * np.diag(mask), 2)
#         solver = pyunlocbox.solvers.gradient_descent(step=step)
#         x0 = y1.copy()
#         prob2 = pyunlocbox.solvers.solve([rr, ff], solver=solver,x0=x0, rtol=0, maxit=1000,verbosity='NONE')
#         mse_t1 += (np.square(original_signal - prob1['sol'])).mean()
#         mse_t2 += (np.square(original_signal - original_signal2)).mean()
#         mse_t3 += (np.square(original_signal - prob2['sol'])).mean()
#         mae_t1 += np.sum(np.absolute(original_signal - prob1['sol']))
#         mae_t2 += np.sum(np.absolute(original_signal - original_signal2))
#         mae_t3 += np.sum(np.absolute(original_signal - prob2['sol']))

#         counter += 1
#     mse[0,sigm] = mse_t1 / counter
#     mse[1,sigm] = mse_t2 / counter
#     mse[2,sigm] = mse_t3 / counter
#     mae[0,sigm] = mae_t1 / counter
#     mae[1,sigm] = mae_t2 / counter
#     mae[2,sigm] = mae_t3 / counter
#     mse_t1, mse_t2, mse_t3 = 0,0,0
#     mae_t1, mae_t2, mae_t3 = 0,0,0
#     counter = 0

# fig,ax = plt.subplots(2,1)
# ax[0].plot(np.arange(-10,45,5),mse[0,:],label="Total")
# ax[0].plot(np.arange(-10,45,5),mse[1,:],label="Proposed")
# ax[0].plot(np.arange(-10,45,5),mse[2,:],label="Laplacian")
# ax[1].plot(np.arange(-10,45,5),mae[0,:],label="Total")
# ax[1].plot(np.arange(-10,45,5),mae[1,:],label="Proposed")
# ax[1].plot(np.arange(-10,45,5),mae[2,:],label="Laplacian")
# ax[0].set_yscale("log")
# ax[1].set_yscale("log")
# ax[0].grid(True)
# ax[1].grid(True)
# plt.show()



def find_branches(H_in,expo,debug=False,rc=True):
    if expo == 0:
        return 1,0,0
    cc = H_in.copy()
    if expo > 1:
        np.fill_diagonal(cc,0)
    n,m = np.shape(cc)
    temp_c = (cc**expo)
    vec_tc = temp_c.reshape((-1,1),order="F")
    sigma = np.where(vec_tc == 1)
    I = np.identity(m)
    out = out = np.zeros((n,int(np.asscalar(np.sum(vec_tc[sigma[0]],axis=0)))))
    rows = np.mod(sigma, n)[0]
    cols = np.ceil(sigma[0]/n).astype(int)
    rows[rows==0] = m 
    flag = 0
    counter = 0
    for k in range(len(rows)):
        if cols[k] == flag:
            out[rows[k],k] = 1
            counter += 1
        else:
            out[rows[k],k] = 1
        flag = cols[k]
    if expo == 1:
        count = 0
        for tt,t in enumerate(cols):
            out[:,count] += I[:,t-1]
            count += 1

    else:
        o1,r1,c1 = find_branches(cc, expo-1,False,False) 
        for jj in range(len(cols)):
            sC, sR = cols[jj], rows[jj]
            preselect = np.where(cc.T[:,sR] == 1)
            hits = [item for x in [np.where(r1==preselect[0][k])[0] for k in range(len(preselect[0]))] for item in x ]
            for k in hits:
                if c1[k] == sC:
                    out[:,jj] += o1[:,k]
                else:
                    continue

    return out,rows,cols
#find_branches()

def find_rounting(D):
    print("Calculating dictionary matrix ... if number of vertices above 500 take some coffee!")
    n,m = np.shape(D)
    Dp = -np.minimum(D.todense(), np.zeros((n,m)))
    Dm = np.maximum(D.todense(), np.zeros((n,m)))
    H = np.dot(Dp.T,Dm)
    memory = np.zeros((n-1,1))
    for k in range(n-1):
        H_p = (np.linalg.matrix_power(H,k)).reshape((-1,1),order="F")
        temp =  np.where(H_p == 1)
        if not any(temp[0]):
            break
        memory[k] = np.sum(H_p[temp])
    R_width = np.sum(memory) + m 
    R = np.zeros((m,int(R_width)))
    R[0:m,0:m] = np.identity(m)
    memory = memory[memory != 0]
    start = m-1
    for j in range(1,len(memory)):
        temp, _, _ = find_branches(H,j,False,False)
        R[:,int(start+1):int(start+1+memory[j])] = temp
        start += memory[j]
    redundant = np.squeeze(np.asarray(np.sum(np.dot(Dp,R) > 1 ,axis=0) + np.sum(np.dot(Dm,R) > 1, axis=0)))
    R_nonred = np.zeros((m,np.sum(redundant==0)))
    counter = 0
    for k in range(int(R_width)):
        if redundant[k] == 0:
            R_nonred[:,counter] = R[:,k]
            counter += 1

    return R_nonred

#mobility_pattern = ["137_D1113","137_D111","137_E4","170_D111","170_E3","169_D111","147_D112","147_E2","142_D41","142_E3","144_D41",
#                    "144_E2","146_D42","146_E2","70_D11","70_E4","69_V51","69_E4","90_V51","90_E4"]    
#mobility_pattern = ["97_D121.1","97_E1","97_D81","97_D51"]    

#mob_sig = [k for k,i in enumerate(Darmstadt.nodes().keys()) if i in mobility_pattern]
#ground_truth_mob_vertex = np.zeros(D_g.n_vertices)
#ground_truth_mob_vertex[mob_sig] = 1
#D_g.plot(vertex_color='#a142f5',highlight=mob_sig)
#plt.show()
#RR = find_rounting(DD)
#R_diag = np.diag(np.dot(RR.T,RR))
#RR2 = RR[:,R_diag != 0]
RR2 = np.load("NonRedundantMatrix2.npy")
#ground_truth_mob_edge=np.squeeze(np.asarray(np.sum(Dp[mob_sig],axis=0).T))
#y_edge = ground_truth_mob_edge + np.random.normal(0,0.01,D_g.n_edges)
#mu = 0.2*np.linalg.norm(np.dot(y_edge,RR2),np.inf)

"""
def s_t_graph(Graph,show=False):
    fig,ax = plt.subplots(1,1)
    GGG = Graph.copy()
    GGG.add_node("s",pos=[8.638,49.8665])
    GGG.add_node("t",pos=[8.652,49.877])
    #GGG._node['s'] = {'x':8.638,'y':49.8665}
    #GGG._node['t'] = {'x':8.652,'y':49.877}
    for k in GGG.nodes():
            GGG.add_edge("s", k)
            GGG.add_edge(k, "t")
    GGG.remove_edge("s","t")
    GGG.remove_edge("s","s")
    GGG.remove_edge("t","t")
    #GGG.remove_edge("t","t")
    # xxs = [x for _, x in GGG.nodes(data='x')]
    # yys = [y for _, y in GGG.nodes(data='y')]
    # xyids = [ID for ID,_ in GGG.nodes(data='osmid')]
    # posi2 = dict(zip(xyids,zip(xxs,yys)))
    # #labels_nodes = dict(zip(GGG.nodes(),range(GGG.number_of_nodes())))
    # labels_nodes = dict()
    # labels_nodes['s'] = 's'
    # labels_nodes['t'] = 't'
    if show:
        nx.draw_networkx_nodes(GGG, pos=posi2, nodelist=GGG.nodes(),node_color="#a142f5",node_edge_color="black", with_labels=False, node_size=30, edge_color="gray",edge_linewidth=3,edge_alpha=0.5,ax=ax)
        nx.draw_networkx_labels(GGG, pos=posi2, labels=labels_nodes, font_size=18,ax=ax)
        nx.draw_networkx_nodes(GGG, pos=posi2, nodelist=['s','t'], node_color=["red","blue"], with_labels=False, node_size=50, node_edge_color="black",ax=ax)
        nx.draw_networkx_edges(GGG, pos=posi2, edgelist=[k for k in GGG.edges(data='name') if k[2] != None],edge_color="gray", alpha=0.5, arrows=True,ax=ax)
        col = nx.draw_networkx_edges(GGG, pos=posi2, edgelist=[k for k in GGG.edges(data='name') if (k[2] == None and k[0] == 's')], edge_color="red", alpha=0.2, arrowsize=20,ax=ax)
        for patch in col:
            patch.set_linestyle('dotted')
        col2 = nx.draw_networkx_edges(GGG, pos=posi2, edgelist=[k for k in GGG.edges(data='name') if (k[2] == None and k[1] == 't')], edge_color="blue", alpha=0.2, arrowsize=20,ax=ax)
        for patch in col2:
            patch.set_linestyle('dotted')
        plt.margins(x=-0.18,y=-0.2)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_title("ST Graph of Darmstadt City")
        plt.tight_layout()
        plt.show()
    else:
        return GGG

def boykov_kolmogorov_maxcut(y,st_Graph):
    count = 0
    for k in st_Graph.nodes():
        if k != None and count <= (len(st_Graph.nodes())-3):
            #st_Graph.nodes()[k[1]]['capacity'] = y[count]
            st_Graph.nodes()[k]['capacity'] = y[count]
            count += 1
    for k in list(st_Graph.edges()):
        if k[0] != 's' and k[1] != 't':
            #st_Graph.edges()._adjdict[k[0]][k[1]][0]['capacity'] = np.mean(y)
            st_Graph.edges()._adjdict[k[0]][k[1]]['capacity'] = np.mean(y)
        elif k[0] == 's':
            #st_Graph.edges()._adjdict[k[0]][k[1]][0]['capacity'] = np.mean(y)*st_Graph.nodes()[k[1]]['capacity']
            st_Graph.edges()._adjdict[k[0]][k[1]]['capacity'] = np.mean(y)*st_Graph.nodes()[k[1]]['capacity']
        elif k[1] == 't':
            #st_Graph.edges()._adjdict[k[0]][k[1]][0]['capacity'] =  np.mean(y)*(1 - st_Graph.nodes()[k[0]]['capacity'])
            st_Graph.edges()._adjdict[k[0]][k[1]]['capacity'] =  np.mean(y)*(1 - st_Graph.nodes()[k[0]]['capacity'])
    R = nx.algorithms.flow.boykov_kolmogorov(nx.DiGraph(st_Graph),'s','t')
    lookup = dict(zip(st_Graph.nodes(),range(st_Graph.number_of_nodes())))
    results = []
    for k in list(R.graph['trees'][0].keys()):
        if k != 's':
            results.append(lookup[k])
    flow_value = R.graph['flow_value']
    return results

def path_based(Graph, signals, lambd,verbose=False, threshold=True):
    Adjacency = Graph.A.astype(np.double)
    variable = cp.Variable(Graph.n_vertices, nonneg=True)
    constraint = [cp.norm(Adjacency @ variable, 'inf') <= 2, variable <= 1]
    object_path = cp.Minimize(cp.sum_squares(signals - variable) + lambd * cp.norm(variable, 1))
    problem = cp.Problem(object_path, constraint)
    problem.solve(solver=cp.GUROBI,verbose=verbose)
    if threshold:
        variable.value[variable.value <= lambd] = 0
        variable.value[variable.value != 0] = 1
    if len(variable.value.nonzero()[0]) != 0:
        W_ = Adjacency.copy()
        rows, cols = [k for k in W_.nonzero()[0]], [k for k in W_.nonzero()[1]]
        W = W_[np.ix_(variable.value.nonzero()[0], variable.value.nonzero()[0])]
        subAdjacency = Adjacency[np.ix_(variable.value.nonzero()[0],variable.value.nonzero()[0])]
        sparse_var = cp.Variable(subAdjacency.shape[0], boolean=True)
        constraint2 = [2*cp.sum(sparse_var) - cp.sum(subAdjacency @ sparse_var) == 2]
        object_path_sparse = cp.Minimize(cp.norm(signals[np.ix_(variable.value.nonzero()[0])] - sparse_var) - np.ones(subAdjacency.shape[0],)@sparse_var)
        subProblem = cp.Problem(object_path_sparse, constraint2)
        subProblem.solve(solver=cp.GUROBI, warmstart=True, verbose=verbose)
        # reconstruct = dict(zip(self.variable.value.nonzero()[0],self.subProblem.value))
        # print(reconstruct)
        #print("Anzahl der aktivierte Knoten {0}  Anzahl der optimierten Knoten {1}".format(len(self.variable.value.nonzero()[0]), len(self.sparse_var.value.nonzero()[0])))
    return variable.value

def glap_binary(Graph, signals, lambd, verbose=False):
    variable = cp.Variable(Graph.n_vertices, boolean=True)
    obje = cp.Minimize(cp.sum_squares(signals-variable) + lambd * cp.quad_form(variable,Graph.L))
    problem = cp.Problem(obje)
    problem.solve(solver=cp.GUROBI, warmstart=True,verbose=verbose)
    return variable.value

def trend_filtering(Graph, signals, lambd, verbose=False):
    variable = cp.Variable(Graph.n_vertices, boolean=True)
    obje = cp.Minimize(cp.sum_squares(signals-variable) + lambd * cp.norm(Graph.D.T.toarray()*variable,1))
    problem = cp.Problem(obje)
    problem.solve(solver=cp.GUROBI, warmstart=True,verbose=verbose)
    return variable.value

y_edge = ground_truth_mob_edge + np.random.normal(0,0.1,D_g.n_edges)
y_noise = np.squeeze(np.asarray((np.dot(Dm,y_edge))))


st_graph = s_t_graph(Darmstadt,show=False)


import stela 
metrics = np.zeros((15,11))

mse_t, hamming_t, f1_t = 0,0,0
mse_boy, hamming_boy, f1_boy = 0,0,0
mse_path, hamming_path, f1_path = 0,0,0
mse_lap, hamming_lap, f1_lap = 0,0,0
mse_trend, hamming_trend, f1_trend = 0,0,0

noise_level = 1/(10**(np.arange(-10,45,5)/20))
counter = 0
g_constant = np.sum(ground_truth_mob_vertex)
for sigm in range(11):
    sigma = noise_level[sigm]
    print("Sigma level {}".format(sigma))
    for t in tqdm.trange(100):
        y_edge = ground_truth_mob_edge + np.random.normal(0,sigma,D_g.n_edges)
        y_noise = np.squeeze(np.asarray((np.dot(Dm,y_edge))))
        mu = 0.2*np.linalg.norm(np.dot(y_edge,RR2),np.inf)
        
        # Optimization
        obj1, x1, err1 = stela.stela_lasso(RR2,y_edge,mu,1000,verbosity=False)
        x_boy = boykov_kolmogorov_maxcut(y_noise, st_graph)
        x_pathbased = path_based(D_g,y_noise,0.42,False,True) 
        x_lap = glap_binary(D_g,y_noise,0.3,False)
        x_trend = trend_filtering(D_g,y_noise,0.42,False)

        # Transformations
        rec_mob = np.squeeze(np.asarray(np.dot(Dp,np.sum(RR2[:,x1.argsort()[-3:][::-1]],axis=1)))) 
        rec_choose = np.zeros(D_g.n_vertices)
        rec_choose[rec_mob!=0] = 1
        x_boykov = np.zeros(D_g.n_vertices)
        x_boykov[x_boy] = 1

        mse_t += np.square(ground_truth_mob_vertex - rec_mob).mean()
        hamming_t += np.sum(np.absolute(ground_truth_mob_vertex - rec_mob))
        f1_t += 2* np.sum(ground_truth_mob_vertex * rec_choose) / (g_constant + np.sum(rec_choose))

        mse_boy += np.square(ground_truth_mob_vertex - x_boykov).mean()
        hamming_boy += np.sum(np.absolute(ground_truth_mob_vertex - x_boykov))
        f1_boy += 2* np.sum(ground_truth_mob_vertex * x_boykov) / (g_constant + np.sum(x_boykov))

        mse_path += np.square(ground_truth_mob_vertex - x_pathbased).mean()
        hamming_path += np.sum(np.absolute(ground_truth_mob_vertex - x_pathbased))
        f1_path += 2* np.sum(ground_truth_mob_vertex * x_pathbased) / (g_constant + np.sum(x_pathbased))

        mse_lap += np.square(ground_truth_mob_vertex - x_lap).mean()
        hamming_lap += np.sum(np.absolute(ground_truth_mob_vertex - x_lap))
        f1_lap += 2* np.sum(ground_truth_mob_vertex * x_lap) / (g_constant + np.sum(x_lap))
        
        mse_trend += np.square(ground_truth_mob_vertex - x_trend).mean()
        hamming_trend += np.sum(np.absolute(ground_truth_mob_vertex - x_trend))
        f1_trend += 2* np.sum(ground_truth_mob_vertex * x_trend) / (g_constant + np.sum(x_trend))
        
        counter += 1
    metrics[0,sigm] = mse_t / counter
    metrics[1,sigm] = hamming_t / counter
    metrics[2,sigm] = f1_t / counter
    metrics[3,sigm] = mse_boy / counter
    metrics[4,sigm] = hamming_boy / counter
    metrics[5,sigm] = f1_boy / counter
    metrics[6,sigm] = mse_path / counter
    metrics[7,sigm] = hamming_path / counter
    metrics[8,sigm] = f1_path / counter
    metrics[9,sigm] = mse_lap / counter
    metrics[10,sigm] = hamming_lap / counter
    metrics[11,sigm] = f1_lap / counter
    metrics[12,sigm] = mse_trend / counter
    metrics[13,sigm] = hamming_trend / counter
    metrics[14,sigm] = f1_trend / counter
    mse_t,hamming_t,f1_t = 0,0,0
    mse_boy,hamming_boy,f1_boy = 0,0,0
    mse_path,hamming_path,f1_path = 0,0,0
    mse_lap,hamming_lap,f1_lap = 0,0,0
    mse_trend,hamming_trend,f1_trend = 0,0,0
    counter = 0
np.save("ErgebnisseBall.npy",metrics)
metrics = np.load("ErgebnisseBAll.npy")
fig,ax = plt.subplots(1,1)
ax.plot(np.arange(-10,45,5),metrics[0,:],marker=".",label="Proposed")
ax.plot(np.arange(-10,45,5),metrics[3,:],marker="x",label="Boykov")
ax.plot(np.arange(-10,45,5),metrics[6,:],marker="o",label="Path-based")
ax.plot(np.arange(-10,45,5),metrics[9,:],marker="s",label="Laplacian")
ax.plot(np.arange(-10,45,5),metrics[12,:],marker="v",label="Trend")
ax.set_xlabel('SNR in dB')
ax.set_ylabel('MSE')
ax.grid(True)
plt.show()
fig,ax = plt.subplots(1,1)
ax.plot(np.arange(-10,45,5),metrics[1,:],marker=".",label="Proposed")
ax.plot(np.arange(-10,45,5),metrics[4,:],marker="x",label="Boykov")
ax.plot(np.arange(-10,45,5),metrics[7,:],marker="o",label="Path-based")
ax.plot(np.arange(-10,45,5),metrics[10,:],marker="s",label="Laplacian")
ax.plot(np.arange(-10,45,5),metrics[13,:],marker="v",label="Trend")
ax.set_xlabel('SNR in dB')
ax.set_ylabel('Hamming Distance')
ax.grid(True)
plt.show()
fig,ax = plt.subplots(1,1)
ax.plot(np.arange(-10,45,5),metrics[2,:],marker=".",label="Proposed")
ax.plot(np.arange(-10,45,5),metrics[5,:],marker="x",label="Boykov")
ax.plot(np.arange(-10,45,5),metrics[8,:],marker="o",label="Path-based")
ax.plot(np.arange(-10,45,5),metrics[11,:],marker="o",label="Laplacian")
ax.plot(np.arange(-10,45,5),metrics[14,:],marker="v",label="Trend")
ax.set_xlabel('SNR in dB')
ax.set_ylabel('F1 Score')
ax.grid(True)
plt.show()

"""
import time

def network_stela(Y, R, rho, maxiter=100):
    NN,II = np.shape(R)
    _,KK = np.shape(Y)
    np.random.seed(42)
    val = np.zeros(maxiter+1)
    err = np.zeros(maxiter+1)
    CPU_time = np.zeros(maxiter + 1)
    CPU_time[0] = time.time()
    lambd = 10 * 0.001 * np.linalg.norm(Y)
    mu = 5 * 0.01 * np.linalg.norm(np.dot(R.T,Y),np.Inf)
    initial_P = np.random.randn(NN,rho)#np.sqrt(100/II) * 
    initial_Q =  np.random.randn(rho,KK)#np.sqrt(100/KK) *
    initial_S = np.zeros((II,KK))
    val[0] = 0.5 * np.linalg.norm(Y - np.dot(initial_P,initial_Q))**2 + 0.5* lambd * (np.linalg.norm(initial_P)**2 + np.linalg.norm(initial_Q)**2)

    P = initial_P
    Q = initial_Q
    S = initial_S
    zeroIK = np.zeros((II,KK))
    onesIK = np.ones((II,KK))
    r_RtR = np.sum(np.multiply(R, R), axis=0)
    epsilon = np.dot(P,Q) + np.dot(R,S) - Y
    err[0] = np.abs(np.trace(np.dot(P.T, np.dot(epsilon,Q.T) + lambd * P)) + np.trace(np.dot(Q.T, np.dot(P.T,epsilon) + lambd * Q)) 
            + np.trace(np.dot(S.T, np.dot(R.T,epsilon))) )

    IterationOutput = "{0:9}|{1:10}|{2:15}|{3:15}|{4:15}"
    print(IterationOutput.format("Iteration", "stepsize", "objval", "error", "CPU time"))
    print(IterationOutput.format(0, 'N/A', format(val[0], '.7f'), format(err[0], '.7f'), format(CPU_time[0], '.7f')))

    for t in range(maxiter):
        Y_DS = Y - np.dot(R,S)
        P_new = np.dot(np.dot(Y_DS,Q.T) , np.linalg.inv(np.dot(Q,Q.T) + lambd * np.identity(rho)))
        cp = P_new - P

        Q_new = np.dot(np.linalg.inv(np.dot(P.T,P) + lambd * np.identity(rho)) , np.dot(P.T, Y_DS))
        cq = Q_new - Q 

        #G = np.dot(np.diag(r_RtR),S) - np.dot(R.T, np.dot(P,Q) - Y_DS)
        G = S - np.dot(R.T, np.dot(P,Q)-Y_DS) / r_RtR[:,None]
        #S_new = np.dot(np.linalg.inv(np.diag(r_RtR)), np.maximum(G - mu*onesIK,zeroIK) - np.maximum(-G - mu*onesIK,zeroIK))
        S_new = np.maximum(G - mu*onesIK,zeroIK) - np.maximum(-G - mu*onesIK,zeroIK)
        cs = S_new - S

        A = np.dot(cp, cq)
        B = np.dot(P, cq) + np.dot(cp,Q) + np.dot(R,cs)
        C = np.dot(P,Q) + np.dot(R,S) - Y

        a = 2 * np.sum(np.sum(A**2,axis=0))
        b = 3 * np.sum(np.sum(A*B,axis=0))
        c = np.sum(np.sum(B**2,axis=0)) + 2*np.sum(np.sum(A*C,axis=0)) + lambd*(np.sum(np.sum(cp**2,axis=0)) + np.sum(np.sum(cq**2,axis=0)))
        d = np.sum(np.sum(B*C,axis=0)) + lambd * (np.sum(np.sum(cp*P, axis=0)) + np.sum(np.sum(cq*Q,axis=0))) + mu * (np.linalg.norm(S_new,1) - np.linalg.norm(S,1))

        sigma1 = (-(b/3/a)**3 + b*c/6/(a**2) - d/2/a)
        sigma2 = c/3/a - (b/3/a)**2
        sigma3 = sigma1**2 + sigma2**3
        sigma3sqrt = np.sqrt(sigma3)

        if sigma3 >= 0:
            gamma = np.cbrt(sigma1 + sigma3sqrt) + np.cbrt(sigma1 - sigma3sqrt) - b/3/a
        else:
            c1 = [1, 0, 0, -(sigma1 + sigma3sqrt)]
            c2 = [1, 0, 0, -(sigma1 - sigma3sqrt)]
            R = np.roots(c1).real + np.roots(c2).real - b/3/a * np.ones(3)
            gamma = np.minimum(R[R>0])
        gamma = np.maximum(0,np.minimum(gamma,1))
        epsilon += gamma * B + gamma**2 * A
        err[t+1] = np.abs(np.trace(np.dot(cp.T, np.dot(epsilon,Q.T) + lambd * P)) + np.trace(np.dot(cq.T, np.dot(P.T,epsilon) + lambd * Q)) 
            + np.trace(np.dot(cs.T, np.dot(R.T,epsilon))) + mu * (np.linalg.norm(cs,1) - np.linalg.norm(S,1)) )
        
        P += gamma * cp 
        Q += gamma * cq
        S += gamma * cs 
        CPU_time[t + 1] = time.time() - CPU_time[t + 1] + CPU_time[t]
        val[t+1] = 0.5 * np.linalg.norm(Y - np.dot(P,Q) - np.dot(R,S))**2 + 0.5* lambd * (np.linalg.norm(P)**2 + np.linalg.norm(Q)**2) + mu * np.linalg.norm(S,1)
        print(IterationOutput.format(t, gamma, format(val[t+1], '.7f'), format(err[t+1], '.7f'), format(CPU_time[t+1], '.7f')))
        if err[t+1] <= 1:
            print("Optimum reached")
            print("check optimality of solution: {} lambda {}".format(np.linalg.norm(Y - np.dot(P,Q) - np.dot(R,S)),lambd))
            return P,Q,S,val,err
        
    print("check optimality of solution: {} lambda {}".format(np.linalg.norm(Y - np.dot(P,Q) - np.dot(R,S)),lambd))
            
    return P,Q,S,val,err


P,Q,S,val,err = network_stela(XX-SS, RR2, 10, 20)
im = plt.imshow(S,cmap='purples ')
np.shape(P)



    
# def sigplot():
#     fig,ax=plt.subplots(3,1)
#     ax[0].plot(original_signal,marker="o",color='#a142f5',linestyle=":",label=r'$\mathbf{y}_{original}$')
#     ax[0].plot(prob2['sol'],marker=".",color="green",label=r'$\hat{\mathbf{y}}_{TV} $')
#     ax[1].plot(original_signal,marker="o",color='#a142f5',linestyle=":",label=r'$\mathbf{y}_{original}$')
#     ax[1].plot(original_signal2,marker="x",color="gold",label=r'$\hat{\mathbf{y}}_{prop}$')
#     ax[2].plot(original_signal,marker="o",color='#a142f5',linestyle=":",label=r'$\mathbf{y}_{original}$')
#     ax[2].plot(prob1['sol'],marker="s",color="red",label=r'$\hat{\mathbf{y}}_{Lap}$')
#     #ax[0].plot(np.squeeze(np.asarray(np.dot(Dp,x.value-s.value))),marker=".",label=r'$\mathbf{D}_x \mathbf{x}-\mathbf{s}$')
#     #ax[0].plot(np.squeeze(np.asarray(np.dot(-Dp,s.value*silent))),marker="s",label=r'$\mathbf{D}_x \mathbf{x}')
#     #plt.plot(np.squeeze(np.asarray(np.dot(Dp,x.value+s.value))),marker="v")
#     #plt.show()
#     #ax[1].plot(x.value,marker="x")
#     #ax[1].plot(s.value,marker=".")
#     ax[0].spines['right'].set_visible(False)
#     ax[0].spines['top'].set_visible(False)
#     ax[0].spines['bottom'].set_visible(False)
#     ax[1].spines['right'].set_visible(False)
#     ax[1].spines['top'].set_visible(False)
#     ax[1].spines['bottom'].set_visible(False)
#     ax[2].spines['right'].set_visible(False)
#     ax[2].spines['top'].set_visible(False)
#     for k,i in enumerate(mask):
#         if i == 0:
#             ax[0].axvline(k, ymin=0,ymax=14,color="black",linestyle="-.",alpha=0.3)
#             ax[1].axvline(k, ymin=0,ymax=14,color="black",linestyle="-.",alpha=0.3)
#             ax[2].axvline(k, ymin=0,ymax=14,color="black",linestyle="-.",alpha=0.3)
#     ax[0].set_xlim(0,233)
#     ax[1].set_xlim(0,233)
#     ax[2].set_xlim(0,233)
#     ax[2].set_xlabel("Vertex index k")
#     ax[1].set_ylabel("Measured cars")
#     ax[0].legend()
#     ax[1].legend()
#     ax[2].legend()
#     fig.suptitle("Comparison graph reconstruction")
#     fig.tight_layout()
#     plt.show()

# #

# sigplot()

# fig,ax = plt.subplots(2,1)
# D_g.plot(original_signal,vertex_size=30,ax=ax[0],title="Ground truth signal")
# ax[0].set_axis_off()
# ax[1].plot(original_signal,marker="x",color='#a142f5',label=r'$\mathbf{y}_{original}$')
# ax[1].spines['right'].set_visible(False)
# ax[1].spines['top'].set_visible(False)
# plt.legend()
# plt.tight_layout()
# plt.show()

"""
G = nx.DiGraph()
G.add_node("0",pos=[4,8])
G.add_node("1",pos=[8.5,8])
G.add_node("2",pos=[5,6.5])
G.add_node("3",pos=[1.5,5])
G.add_node("4",pos=[8,4])
G.add_node("5",pos=[9,1])
G.add_node("6",pos=[0.5,1])
G.add_node("7",pos=[4.6,0.5])

pos = nx.get_node_attributes(G,'pos')
G.add_edge("3","0",weight=0.24)
G.add_edge("0","1",weight=0.23)
G.add_edge("2","0",weight=0.74)
G.add_edge("1","2",weight=0.35)
G.add_edge("2","3",weight=0.26)
G.add_edge("6","3",weight=0.32)
G.add_edge("6","7",weight=0.32)
G.add_edge("7","6",weight=0.32)
G.add_edge("5","7",weight=0.15)
G.add_edge("4","5",weight=0.51)
G.add_edge("4","1",weight=0.23)
G.add_edge("2","4",weight=0.24)
G.add_edge("4","2",weight=0.24)
G.add_edge("7","4",weight=0.14)
G.add_edge("4","7",weight=0.14)
labels = nx.get_edge_attributes(G,'weights')

fig,ax = plt.subplots()
nx.draw_networkx_nodes(G,pos=pos,node_color='#a142f5',ax=ax)
nx.draw_networkx_labels(G,pos=pos,font_size=14,ax=ax)
nx.draw_networkx_edges(G,pos=pos,ax=ax)
nx.draw_networkx_edge_labels(G,pos=pos,edge_labels=labels,ax=ax)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.plot([4,4],[8,10.3],color='r') #2.3
plt.plot([8.5,8.5],[8,6.3],color='r') #-1.7
plt.plot([5,5],[6.5,8.8],color='r') #2.3
plt.plot([8,8],[4,1.3],color='r') #-2.7
plt.plot([1.5,1.5],[5,7.5],color='r') # 2.5
plt.plot([9,9],[1,3.7],color='r') # 2.7
plt.plot([0.5,0.5],[1,3.5],color='r') # 2.5
plt.plot([4.6,4.6],[0.5,-1.5],color='r') # -2
plt.scatter(4,10.3,color='r')
plt.scatter(8.5,6.3,color='r')
plt.scatter(5,8.8,color='r')
plt.scatter(1.5,7.5,color='r')
plt.scatter(8,1.3,color='r')
plt.scatter(9,3.7,color='r')
plt.scatter(0.5,3.5,color='r')
plt.scatter(4.6,-1.5,color='r')
plt.show()

G_s = ps.graphs.Graph(nx.adjacency_matrix(G))
G_s.set_coordinates([v for v in nx.get_node_attributes(G,'pos').values()])
signal = [2.3,-1.7,2.3,-2.7,2.5,2.7,2.5,-2]
shift_sig = [-0.391, 0.805, 1.6, 0.522, 1.258, -0.3, -1.504, 1.15]
fig2,ax2 = plt.subplots(1,3) 
G_s.plot(signal,limits=[-2.7,2.7],ax=ax2[0],vertex_size=100,colorbar=False)
G_s.plot(shift_sig,limits=[-2.7,2.7],ax=ax2[1],vertex_size=100,colorbar=False)
G_s.plot((nx.adjacency_matrix(G)**2)*signal,limits=[-2.7,2.7],ax=ax2[2],vertex_size=100,colorbar=False)
cax = fig2.add_axes([0.1, 0.05, 0.8, 0.01])
fig2.subplots_adjust(hspace=0.5,right=0.8)
ax2[0].set_axis_off()
ax2[1].set_axis_off()
ax2[2].set_axis_off()
ax2[0].set_title(r'$\mathbf{y}$')
ax2[1].set_title(r'$\mathbf{A}\mathbf{y}$')
ax2[2].set_title(r'$\mathbf{A}^2\mathbf{y}$')
cbar = fig2.colorbar(ax2[0].collections[0],cax=cax,orientation='horizontal')
cbar.minorticks_on()
cbar.ax.set_xticklabels([-2.7,-1.62,-0.54,0.54,1.62,2.7])
plt.show()

G_s.compute_laplacian('normalized')
fig2,ax2 = plt.subplots(1,1)
G_s.plot(eigenvalues=True,ax=ax2)
plt.show()


fig2,ax2=plt.subplots(2,4)
signal2 = signal# / np.linalg.norm(signal)
shift_sig = (nx.adjacency_matrix(G)*signal2) #/ np.linalg.norm(nx.adjacency_matrix(G)*signal2)
doubleshift = (nx.adjacency_matrix(G)**2)*signal2 #/ np.linalg.norm((nx.adjacency_matrix(G)**2)*signal2)
G_s.plot(signal2,ax=ax2[0,0],vertex_size=100)
x = G_s.gft(signal2)
ax2[0,0].set_title(r'$x^T L x = {:.2f}$'.format(G_s.dirichlet_energy(x)))
ax2[0,0].set_axis_off()
ax2[1,0].plot(G_s.e, np.abs(x), '.-')
ax2[1,0].set_xlabel(r'graph frequency $\lambda$')

G_s.plot(shift_sig,ax=ax2[0,1],vertex_size=100)
x = G_s.gft(shift_sig)
ax2[0,1].set_title(r'$x^T L x = {:.2f}$'.format(G_s.dirichlet_energy(x)))
ax2[0,1].set_axis_off()
ax2[1,1].plot(G_s.e, np.abs(x), '.-')
ax2[1,1].set_xlabel(r'graph frequency $\lambda$')

G_s.plot(doubleshift,ax=ax2[0,2],vertex_size=100)
x = G_s.gft(doubleshift)
ax2[0,2].set_title(r'$x^T L x = {:.2f}$'.format(G_s.dirichlet_energy(x)))
ax2[0,2].set_axis_off()
ax2[1,2].plot(G_s.e, np.abs(x), '.-')
ax2[1,2].set_xlabel(r'graph frequency $\lambda$')


plt.show()

"""


