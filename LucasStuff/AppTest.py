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

# Defining the models we have
models = ['Linear Regression / GLM', 'Logistic Regression', 'K-Means', 'KNN', 'PCA']

# Geojson is called a "shape file", represents shape on a map
r = requests.get('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json')
geojson = json.loads(r.text)

#####################################################################
###               APP GRAPHICS + LAYOUT                           ###
#####################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Modeling Tundra Tree Growth -- DS 6021 Final Project Dashboard'),
    dcc.Markdown('Final project for DS 6021, modeling the growth of trees in \
    the Canadian and Alaskan Tundra'),

    dcc.Tabs([

        # ------------------------------------------------------- #
        #                        SUMMARY TAB                      #
        # ------------------------------------------------------- #
        dcc.Tab(label='Summary', children=[
            html.Div([
                html.H2("Summary"),
                html.P("From 2016 to 2019, the Oak Ridge National Laboratory partnered with NASA's Earthdata program\
                to collect 'in situ measurements of radial tree growth' for a population of 50 trees from the white spruce\
                (Picea glauca) and black spruce (Picea mariana) species. Of these 50 trees, 36 were in the Brooks Range of \
                Alaska, and 24 were near Inuvik in the Northwest Territories of Canada. The purpose of this study was to \
                determine the influence of environmental variables like solar irradiance, soil temperature, air pressure, \
                and more on the radial tree growth dynamics."),
                
                html.Hr(),  # optional horizontal separator

        dcc.Markdown(
            """
### **Citations**

**Jensen, J., N. Boelman, J. Eitel, L. Vierling, A.J. Maguire, and K. Griffin.**  
*2023. Dendrometer, Soil, and Weather Observations, Arctic Tree Line, AK and  
NWT, 2016–2019.* ORNL DAAC, Oak Ridge, Tennessee, USA.  
https://doi.org/10.3334/ORNLDAAC/2185
            """,
            style={'paddingTop': '20px'}
        )
            ], style={'padding': '20px'})
        ]),

        # ------------------------------------------------------- #
        #                  EXPLORATORY DATA ANALYSIS TAB          #
        # ------------------------------------------------------- #
        dcc.Tab(label='Exploratory Data Analysis', children=[
            html.Div([
                html.H2("Exploratory Data Analysis"),
                html.P("Below is an exploratory figure:"),

            html.Img(
                src='/assets/TreeRegions.png',
                style={'width': '60%', 'display': 'block', 'margin': 'auto'}
        )
            ], style={'padding': '20px'})
        ]),

        # ------------------------------------------------------- #
        #                        MODELS TAB                       #
        # ------------------------------------------------------- #
        dcc.Tab(label='Models', children=[

            html.Div([
                html.H2("Model Inputs and Outputs"),

                dcc.Dropdown(
                    id='ModelDropdown',
                    options=models,
                    value='Linear Regression / GLM',
                    placeholder="Select a model."
                ),

                html.Div([
                    dcc.Graph(id='Model Parameters'),
                ], style={'width': '60%', 'float': 'left'}),

                html.Div([
                    dcc.Graph(id='countytable')
                ], style={'width': '40%', 'float': 'right'}),

                html.Div(style={'clear': 'both'})  # clears the float
            ], style={'padding': '20px'})
        ])

    ])
])



#####################################################################
###                 FUNCTIONS AND CALLBACKS                       ###
#####################################################################

# Callbacks are listed between the layout and the run method, here:

# @ is a decorator. It means that the following code works on the next function defined in the code

@app.callback(
    Output('countymap', 'figure'),
    Input('countydropdown', 'value')
)
def map_one_county(loc):

    vacounties['loc'] = vacounties['Jurisdiction'] == loc

    fig = px.choropleth(
        vacounties,
        geojson=geojson,
        scope='usa',
        locations='FIPS',
        color='loc',
        hover_name='Jurisdiction',
        hover_data=['Average Annual Pay', 'Cost of living'],
        color_discrete_map={True: 'red', False: 'dodgerblue'}
    )

    fig.update_geos(fitbounds="locations")
    fig.update_layout(showlegend=False)
    return fig


# Update the table when county changes
@app.callback(
    Output('countytable', 'figure'),
    Input('countydropdown', 'value')
)
def countytable(loc):
    vacounties_loc = vacounties[vacounties['Jurisdiction'] == loc]

    table = vacounties_loc.T.reset_index()
    table = table.rename(columns={'index': '', 0: ''})

    return ff.create_table(table)
    
if __name__ == '__main__':
    app.run(debug=True)

