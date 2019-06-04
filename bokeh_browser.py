
from main_file_capped1 import GIST_Algo
import numpy as np 
import holoviews as hv 
from holoviews import opts 
import re
hv.extension('matplotlib','bokeh') 
hv.notebook_extension('matplotlib','bokeh')

# TODO auskommentieren oder anpassen
img = hv.Image(np.random.rand(10,10))
fig = hv.render(img)
hv.output(fig='svg')

# TODO auskommentieren oder anpassen
from holoviews.operation import contours
x = y = np.arange(-3.0, 3.0, 0.1)
X,Y = np.meshgrid(x,y)

# TODO auskommentieren oder anpassen
def g(x,y,c):
    return 2*((x-y)**2/(x**2+y**2)) + np.exp(-(np.sqrt(x**2+y**2)-c)**2)

# TODO auskommentieren oder anpassen
img = hv.Image(g(X,Y,3))
filled_contours = contours(img,filled=True)
gv, ge, gt = GIST_Algo(1000, 1600, 0.01, 1, 400, 0.001, False)
t = np.linspace(0, np.sum(np.transpose(gt)),400) / (10**9)
tt = t
img2 = hv.Curve((t,np.transpose(gv)[:,0]))
hv.save(img2, 'gist.svg')


from bokeh.plotting import figure
from bokeh.models import Button, ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, Select
from bokeh.io import output_notebook, push_notebook, show, output_file, curdoc
from bokeh.layouts import column, row

# TODO backup für html files
#output_file('gist.html')               
plot = figure(title="Gist Algorithmus",toolbar_location=None)
source = ColumnDataSource(data=dict(x=tt, y=np.transpose(gv)[:,0]))
#plot.scatter(t, np.transpose(gv)[:,0])
plot.line('x','y',source=source)
auswahl_dim = {'klein':[200,400],'mittel':[2000,6000],'groß':[10000,20000]}
a_list = []
for k in auswahl_dim:
    a_list.append("{} {}".format(k, auswahl_dim[k]))

dimension = Select(title="Dimension",value='klein',options=a_list)
text = TextInput(title="Offset",value='Zeitverzug')
offset = Slider(title="phase",value=0.0, start=-10.0, end=10.0, step=0.1)
calculate = Button(label="recalculate")
density = Slider(title="densitiy",value=0.01, start=0.01, end=1.0, step=0.01)

def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

o = 0
sigma = 0.01
def update_data(attrname, old, new):
    global o
    o = offset.value
    global tt
    y = np.transpose(gv)[:,0]
    source.data = dict(x=tt+o,y=y)

def rerun_gist(attrname, old, new):
    temp = str(re.findall("\[(.*?)\]",dimension.value))
    temp = temp[2:-2].split(',')
    print(int(temp[0]),int(temp[1]))
    print("Berechne Gist neu")
    global sigma
    gv, ge, gt = GIST_Algo(int(temp[0]), int(temp[1]), sigma, 1, 400, 0.001, False)
    t = np.linspace(0, np.sum(np.transpose(gt)),400) / (10**9)
    global tt
    tt = t
    global o
    y = np.transpose(gv)[:,0]
    source.data = dict(x=tt+o, y=y)


def change_density(attrname, old, new):
    global sigma
    sigma = density.value
    print(sigma)


def rerun_gist_dens():
    temp = str(re.findall("\[(.*?)\]",dimension.value))
    temp = temp[2:-2].split(',')
    print(int(temp[0]),int(temp[1]))
    print("Berechne Gist mit Dichte {}".format(density.value))
    global sigma
    gv, ge, gt = GIST_Algo(int(temp[0]), int(temp[1]), density.value, 1, 400, 0.001, False)
    t = np.linspace(0, np.sum(np.transpose(gt)),400) / (10**9)
    global tt
    tt = t
    global o
    y = np.transpose(gv)[:,0]
    source.data = dict(x=tt+o, y=y)

calculate.on_click(rerun_gist_dens)

for w in [offset]:
    w.on_change('value', update_data)
for k in [dimension]:
    k.on_change('value', rerun_gist)
for l in [density]:
    l.on_change('value', change_density)



inputs = column(text, offset, dimension, density, calculate)
curdoc().add_root(row(inputs,plot))
curdoc().title = "Stela Optimize Framework"