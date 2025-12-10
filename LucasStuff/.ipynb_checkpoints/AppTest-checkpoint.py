import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import requests
import json

from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

#####################################################################
###               ACCESSING TOOLS + DATA                          ###
#####################################################################

df = pd.read_csv('../categorized_data.csv')

# for all columns, if datatype is float, round 2 places
df = df.apply(lambda x: x.round(3) if x.dtype == 'float64' else x)

# Geojson is called a "shape file", represents shape on a map
r = requests.get('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json')
geojson = json.loads(r.text)

#####################################################################
###               APP GRAPHICS + LAYOUT                           ###
#####################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.H1('Virginia Counties Data Dashboard'),
        dcc.Markdown('A project for DS 6001 Module 12'),
        dcc.Tabs(
            [
                dcc.Tab(label = 'Information about one county/city', 
                        children = [
                            dcc.Dropdown(id = 'countydropdown',
                                         options = sorted(vacounties['Jurisdiction']),
                                         value = 'Charlottesville city'),
                            html.Div([
                                dcc.Graph(id = 'countymap') # Step 5: Place output on dashboard
                            ], style={'width':'60%', 'float':'left'}),
                            html.Div([
                                dcc.Graph('countytable')
                            ], style={'width':'40%', 'float':'right'})
                        ]),
                dcc.Tab(label = 'Scatterplots',
                        children=[]),
                dcc.Tab(label = 'Cities vs. Counties',
                        children = [])

            ]
        )


    ]
)


#####################################################################
###                 FUNCTIONS AND CALLBACKS                       ###
#####################################################################

# Callbacks are listed between the layout and the run method, here:

# @ is a decorator. It means that the following code works on the next function defined in the code

# Input is step 2; Output is step 4
@app.callback([Output(component_id = 'countymap', component_property = 'figure')],
              [Input(component_id = 'countydropdown', component_property = 'value')])

# Step 3, create the output 
def map_one_county(loc):
    vacounties['loc'] = vacounties['Jurisdiction'] == loc

    fig = px.choropleth(vacounties,
                     geojson=geojson,
                     scope = 'usa',
                     locations='FIPS',
                     #color='Annual Pay',
                     color='loc',
                     hover_name = 'Jurisdiction',
                     hover_data = ['Average Annual Pay', 'Cost of living'],
                     color_continuous_scale='reds',
                     color_discrete_map={
                         True: 'red',
                         False: 'dodgerblue'
                     }
    )
    fig.update_geos(fitbounds="locations")
    fig.update_layout(showlegend=False)
    return[fig]

@app.callback( [Output(component_id = 'countytable', component_property = 'figure')], 
               [Input(component_id='countydropdown', component_property='figure')])

def countytable(loc):
    #loc = 'Albemarle County'

    vacounties_loc = vacounties.query(f"Jurisdiction == '{loc}'")
    table = vacounties_loc.T.reset_index()
    table = table.rename({'index':'', 1: ''}, axis=1)

    return [ff.create_table(table)]

if __name__ == '__main__':
    app.run(debug=True)

