import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget  # Required for Plotly
from datetime import date
import plotly.graph_objects as go
from sklearn.cluster import KMeans


df = pd.read_csv('categorized_data.csv')
df['Date'] = pd.to_datetime(df['Date']) 

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

df['Date'] = df['Date'].dt.date

# VDP Calc
T = df["average_air_temperature"]
RH = df["average_humidity"]

svp = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
df["VPD"] = svp * (1 - RH / 100)

df['day_of_year'] = pd.to_datetime(df['Date']).dt.dayofyear


# --- Feature Sets ---
available_numeric = [
    'average_humidity', 
    'avg_solar_irradiance', 
    'avg_soil_water_content', 
    'average_air_pressure',
    'average_air_temperature',
    'average_stem_radius',
    'change_stem_radius',
    'average_basal_area',
    'change_basal_area'
]

available_categorical = [
    'freeze_flag', 
    'species', 
    'site', 
    'plot',
    'air_pressure_cat',
    'humidity_cat',
    'solar_cat',
    'soil_moist_cat'
]

knn_available_num = [
    'average_humidity', 
    'avg_solar_irradiance', 
    'avg_soil_water_content', 
    'average_air_pressure',
    'average_air_temperature',
    'average_stem_radius',
    'average_basal_area',
]
knn_available_cat = [
    'freeze_flag', 
    'species', 
    'site'
]

kmeans_default_num = ["average_air_temperature", 
                      "average_humidity", 
                      "avg_solar_irradiance", 
                      "avg_soil_water_content", 
                      "average_air_pressure"]

kmeans_default_cat = ["species", 
                      "plot", 
                      "site"]

# Defaults
default_numeric = ['average_humidity', 'avg_solar_irradiance', 'avg_soil_water_content', 'average_air_pressure']
default_categorical = ['freeze_flag', 'species']

knn_default_num = [
    'average_air_temperature', 
    'average_air_pressure',
    'average_humidity',
    'avg_solar_irradiance', 
    'avg_soil_water_content',
    'average_stem_radius'
]
knn_default_cat = ['species']

kmeans_default_num = ["average_air_temperature", 
                      "average_humidity", 
                      "avg_solar_irradiance", 
                      "avg_soil_water_content", 
                      "average_air_pressure"]

kmeans_default_cat = ["species", 
                      "plot", 
                      "site"]


# --- UI Definition ---
app_ui = ui.page_navbar(
    # Tab 1: Readme
    ui.nav_panel(
        "Readme",
        ui.markdown(
            """
            # Project Overview: Plant Growth Prediction
            
            This application allows researchers to interactively analyze Tree Amplitude outcomes.
            
            ### 1. The Goal
            Our primary objective is to determine which environmental factors—such as soil moisture or solar intensity—are the strongest predictors of tree radius amplitude.

            ### 2. Methodologies
            * **Multiple Regression**: A statistical modeling approach used to quantify how environmental variables influence stem radius amplitude, the size of the daily swelling–shrinking cycle of the tree stem. By modeling predictors like VPD, soil moisture, and solar exposure (and their interactions), the MLR helps identify which conditions most strongly increase or dampen these daily changes in stem size.
            * **Logistic Regression**: A predictive model used to classify change. You can now customize the inputs and explore drivers for specific categories.
            * **Principal Component Analysis (PCA)**: A dimensionality reduction technique used to explore the structure of the data and identify natural clusters.
            * **K-Nearest Neighbors (KNN)**: A supervised machine learning model used to classify growth categories.
            * **K-Means Clustering:** An unsupervised machine learning model used to identify natural clusters in the data.
            """
        )
    ),


    # Tab 2: Data Explorer
    ui.nav_panel(
        "Data Explorer",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Filter Data"),
                ui.input_slider(
                    "date_range", 
                    "Select Date Range:",
                    min=min_date,
                    max=max_date,
                    value=[min_date, max_date], 
                    time_format="%Y-%m-%d"
                ),
                ui.hr(),
                ui.input_selectize("filter_site", "Select Site(s):", choices=[], multiple=True),
                ui.input_selectize("filter_species", "Select Species:", choices=[], multiple=True),
                ui.input_selectize("filter_growth", "Growth Category:", choices=[], multiple=True),
                ui.p(ui.input_action_button("reset_filters", "Reset Filters", class_="btn-sm btn-outline-secondary"))
            ),
            ui.card(
                ui.card_header("Filtered Dataset"),
                ui.output_data_frame("data_table")
            )
        )
    ),

    # Tab 3: Multiple Regression
    ui.nav_panel(
        "Multiple Regression",

        ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Interpretation"),
            ui.markdown("""
            - **Response variable:** `change_stem_radius` (the size of the daily swelling–shrinking cycle of the tree stem)
            - **Higher VPD** → more daytime shrinkage  
            - **More soil water** → slightly larger amplitude  
            - **Interaction negative** → wet soil buffers VPD stress  
            - **VPD formula:** `VPD = SVP * (1 - RH/100)`  where SVP itself is an exponential function of temperature.
            """)
        ),

        # First row: card that used to be the sidebar
        ui.layout_columns(
            ui.card(
                ui.card_header("Multiple Regression (OLS & Mixed Effects)"),
                ui.p(
                    """
                    This section shows how environmental factors, particularly 
                    Vapor Pressure Deficit (VPD), soil volumetric water content (VWC), 
                    and solar irradiance, influence daily radial stem changes 
                    in white spruce at Arctic treeline. Tree dummies in OLS and random intercepts in mixed effects are used to account for the repeated daily measurements per tree, ensuring that baseline differences between trees don’t bias the estimated effects of environmental variables
                    Understanding how VPD, soil water, and solar irradiance affect daily stem changes helps reveal how Arctic treeline trees respond to environmental stress, which can inform predictions about tree growth and ecosystem resilience under climate change.
                    """
                ),
                # ui.hr(),
                # ui.h5("Models Displayed"),
                #ui.tags.ul(
                    #ui.tags.li("1. Mixed Linear Model (random intercept for tree)"),
                    #ui.tags.li("2. OLS with dummy variable for each tree"),
                #),
            ),
            col_widths=(12,)
        ),

        # Second row: model output cards
        ui.layout_columns(
            ui.card(
                ui.card_header("Mixed Effects Model Output"),
                ui.input_action_button("run_mixedlm", "Run Mixed Effects Model"),
                ui.output_text_verbatim(
                    "mixedlm_output",
                    placeholder="Click the button to run mixed effects model"
                )
            ),
            ui.card(
                ui.card_header("OLS Output"),
                ui.input_action_button("run_ols", "Run OLS Model"),
                ui.output_text_verbatim(
                    "ols_output",
                    placeholder="Click the button to run OLS model"
                )
            ),
            col_widths=(6, 6)
        ),
        # ui.hr(),
        # ui.h5("Interpretation"),
        #ui.markdown("""
        #- **Response Variable:** `change_stem_radius` (daily amplitude),
                    
        #- **Higher VPD**→ more daytime shrinkage → larger daily amplitude."),
                    
        #- **More soil water** → trees can swell more → slightly larger amplitude."),
                    
        #- **Interaction negative** → wet soil buffers VPD stress."),
        
        #- **VPD is calculated as VPD** = SVP * (1 - RH/100), where SVP = 0.6108 * exp(17.27 * T / (T + 237.3))"),
    #"""),
            ),
    ),

    # Tab 3: Logistic Regression Model
    ui.nav_panel(
        "Logistic Regression Model",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Model Configuration"),
                ui.p("Select the variables to include in the training model:"),
                
                ui.input_selectize(
                    "model_numeric", 
                    "Numerical Variables:", 
                    choices=available_numeric, 
                    selected=default_numeric, 
                    multiple=True
                ),
                
                ui.input_selectize(
                    "model_categorical", 
                    "Categorical Variables:", 
                    choices=available_categorical, 
                    selected=default_categorical, 
                    multiple=True
                ),
                
                ui.hr(),
                ui.h5("Graph Settings"),
                
                # Dynamic Radio Buttons
                ui.input_radio_buttons(
                    "target_view",
                    "View Predictors For:",
                    choices=["A Lot of Growth", "A Little Growth"], 
                    selected="A Lot of Growth"
                )
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Model Performance Metrics"),
                    ui.h5("Accuracy Score"),
                    ui.output_text_verbatim("accuracy_score"),
                    ui.h5("Detailed Classification Report"),
                    ui.output_text_verbatim("class_report"),
                ),
                ui.card(
                    ui.card_header("Understanding the Model"),
                    
                    ui.h5("Interactive Controls"),
                    ui.p("Use the sidebar on the left to add or remove variables from the model. You can also toggle the graph to see predictors."),

                    ui.h5("What is Logistic Regression?"),
                    ui.p("Unlike linear regression, which predicts a number, Logistic Regression predicts the probability of an event happening (like a large basal area amplitude)."),
                    
                    ui.h5("Understanding the Outputs"),
                    ui.tags.ul(
                        ui.tags.li(ui.strong("Precision"), ": When the model predicts 'Extreme Change', how often is it correct? (Quality)."),
                        ui.tags.li(ui.strong("Recall"), ": Out of all actual 'Extreme Change' cases, how many did the model find? (Quantity)."),
                        ui.tags.li(ui.strong("Coefficients"), ": Positive values (Green) increase the likelihood; Negative values (Red) decrease it.")
                    )
                ),
                col_widths=(5, 7)
            ),
            ui.card(
                ui.card_header("Feature Importance Graph"),
                ui.output_plot("analysis_plot", height="500px")
            )
        )
    ),

    # Tab 4: PCA Analysis
    ui.nav_panel(
        "PCA Analysis",
        # Section 1: Overview
        ui.h3("Overview", style="background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; border-left: 5px solid #007bff;"),
        ui.layout_columns(
            ui.card(
                ui.card_header("1. Feature Correlation Matrix"),
                output_widget("corr_matrix_plot")
            ),
            ui.card(
                ui.card_header("2. Scree Plot -- Variance reduced by adding further components"),
                output_widget("scree_plot")
            ),
            col_widths=(6, 6)
        ),
        # Section 2: Exploring the Principal Components
        ui.h3("Exploring the Principal Components", style="background-color: #f0f0f0; padding: 10px; margin-top: 30px; margin-bottom: 20px; border-left: 5px solid #007bff;"),
        ui.layout_columns(
            ui.card(
                ui.card_header("PCA Plot Settings"),
                ui.input_select(
                    "pca_x_axis",
                    "X-Axis:",
                    choices=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"],
                    selected="PC2"
                ),
                ui.input_select(
                    "pca_y_axis",
                    "Y-Axis:",
                    choices=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"],
                    selected="PC4"
                ),
                ui.input_radio_buttons(
                    "pca_color",
                    "Color By:",
                    choices=["Growth", "Species", "Site"],
                    selected="Growth"
                )
            ),
            ui.card(
                ui.card_header("3. PCA Scatter Plot"),
                output_widget("pca_scatter_plot")
            ),
            col_widths=(3, 9)
        ),
        # Correlation matrix with PCs and explanation
        ui.layout_columns(
            ui.card(
                ui.card_header("4. Correlation Matrix (All Variables + PCs)"),
                output_widget("corr_matrix_with_pcs_plot")
            ),
            ui.card(
                ui.card_header("Understanding This Matrix"),
                ui.div(
                    ui.p(
                        "We use this larger correlation matrix to show how each principal component relates to the original features. This, alongside the loadings plot below, allows us to obtain a better understanding of what these components represent.",
                        style="text-align: center; padding: 40px; font-size: 16px; line-height: 1.6;"
                    ),
                    style="display: flex; align-items: center; justify-content: center; height: 100%;"
                )
            ),
            col_widths=(7, 5)
        ),
        # Loadings plot
        ui.layout_columns(
            ui.card(
                ui.card_header("Loadings Plot Settings"),
                ui.input_select(
                    "loadings_x_axis",
                    "X-Axis:",
                    choices=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"],
                    selected="PC1"
                ),
                ui.input_select(
                    "loadings_y_axis",
                    "Y-Axis:",
                    choices=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"],
                    selected="PC2"
                ),
                ui.input_numeric(
                    "arrow_scale",
                    "Arrow Scale:",
                    value=3,
                    min=0.5,
                    max=10,
                    step=0.5
                )
            ),
            ui.card(
                ui.card_header("5. PCA Loadings Biplot"),
                output_widget("loadings_plot")
            ),
            col_widths=(3, 9)
        ),

        # Section 3: Principal Component Regression
        ui.h3("Principal Component Regression", style="background-color: #f0f0f0; padding: 10px; margin-top: 30px; margin-bottom: 20px; border-left: 5px solid #007bff;"),
        ui.layout_columns(
            ui.card(
                ui.card_header("PCR Metrics"),
                ui.output_text_verbatim("pcr_metrics")
            ),
            ui.card(
                ui.card_header("6. Model Performance vs Components"),
                output_widget("pcr_metrics_plot")
            ),
            col_widths=(5, 7)
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Select Principal Components"),
                ui.p("Choose which PCs to include in the regression:"),
                ui.input_checkbox_group(
                    "pcr_selected_pcs",
                    "Principal Components:",
                    choices=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"],
                    selected=["PC1", "PC2", "PC3"]
                ),
                ui.input_action_button("update_pcr", "Update PCR Model", class_="btn-primary")
            ),
            ui.card(
                ui.card_header("7. PCR Predicted vs Actual"),
                output_widget("pcr_pred_vs_actual")
            ),
            col_widths=(3, 9)
        )
    ),


    # Tab 5: KNN
    ui.nav_panel(
        "KNN",

        # Sidebar: KNN Configuration
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("KNN Configuration"),

                ui.input_action_button("run_knn", "Run KNN Model"),

                ui.p("Select the variables to include in the training model:"),
                
                ui.input_selectize(
                    "knn_model_num", 
                    "Numerical Variables:", 
                    choices=knn_available_num, 
                    selected=knn_default_num, 
                    multiple=True
                ),
                
                ui.input_selectize(
                    "knn_model_cat", 
                    "Categorical Variables:", 
                    choices=knn_available_cat, 
                    selected=knn_default_cat, 
                    multiple=True
                )
            ),

            # Top Card: K Selection plot
            ui.card(
                ui.card_header("Best K Value Selection"),
                output_widget("knn_k_selection_plot")
            ),

            # Middle Cards
            ui.card(
                ui.card_header("Model Performance"),
                
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Metrics"),
                        ui.h6(ui.output_text("knn_accuracy_score")),
                        ui.h6(ui.output_text("knn_b_accuracy_score")),
                        output_widget("knn_confusion_matrix")
                    ),
                    ui.card(
                        ui.card_header("Performance by Category"),
                        output_widget("knn_performance_plot")
                    ),
                    col_widths=(6, 6)
                ),
            ),
            

            # Bottom Card: KNN Scatter Plot
            ui.card(
                ui.card_header("KNN Scatter Plot"),
                ui.input_select("knn_scatter_color_by", "Color by:",
                                  {"target": "True Class", "pred": "Predicted Class", "correct": "Correct / Incorrect"},
                                  selected="target"),
                output_widget("knn_scatter_plot", height="500px")
            )
        )
    ),

    # Tab 6: K-Means Model
    ui.nav_panel(
        "K-Means",

        ui.layout_columns(
            ui.card(
                ui.card_header("K-Means Clustering"),
                ui.markdown(
                    """
                    To understand how different environmental conditions grouped into different clusters -- or what we call **"forests"** --, we used a K-Means algorithm using both numeric and categorical variables.

                    We settled on **three** forests after observing the elbow plot below. Both three and four forests have very similiar silhouette scores and three is the best balance between structure and interpretation.
                    """
                )
            ),
            col_widths=(12,)
        ),


        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("K-Means Configuration"),

                ui.input_action_button("run_kmeans", "Run K-Means Model"),

                ui.p("Select the variables to include in clustering:"),

                ui.input_selectize(
                    "kmeans_model_num",
                    "Numerical Variables:",
                    choices=available_numeric,
                    selected=kmeans_default_num,
                    multiple=True
                ),

                ui.input_selectize(
                    "kmeans_model_cat",
                    "Categorical Variables:",
                    choices=available_categorical,
                    selected=kmeans_default_cat,
                    multiple=True
                ),

                ui.hr(),
                ui.input_slider(
                    "kmeans_k",
                    "Number of clusters (k):",
                    min=2,
                    max=10,
                    value=3,
                    step=1
                ),
            ),

            ui.card(
                ui.card_header("Elbow Plot (Within-Cluster Sum of Squares)"),
                output_widget("kmeans_elbow_plot")
            ),



            ui.card(
                ui.card_header("Forests Visualizations"),
                ui.layout_columns(
                    ui.output_plot("kmeans_temp_humidity"),
                    ui.output_plot("kmeans_solar_soil"),
                    col_widths=(6, 6)
                ),
                ui.layout_columns(
                    ui.output_plot("kmeans_pressure_humidity"),
                    ui.output_plot("kmeans_temp_basal"),
                    col_widths=(6, 6)
                )
            ),

            ui.card(
                ui.card_header("Forests Interpretation (3 Clusters)"),
                ui.tags.table({"class": "table table-sm"},
            
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Forest"),
                        ui.tags.th("Temp"),
                        ui.tags.th("Humidity"),
                        ui.tags.th("Solar"),
                        ui.tags.th("Soil Water"),
                        ui.tags.th("Pressure"),
                        ui.tags.th("Species"),
                        ui.tags.th("Site"),
                        ui.tags.th("Amplitude")
                    )
                ),
                            ui.tags.tbody(
                    # Forest 1
                    ui.tags.tr(
                        ui.tags.td("1"),
                        ui.tags.td(
                            ui.input_text(
                                "forest1_temp",
                                None,
                                placeholder="Hotter"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest1_humidity",
                                None,
                                placeholder="Lower"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest1_solar",
                                None,
                                placeholder="Sunnier"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest1_soil",
                                None,
                                placeholder="Wetter"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest1_pressure",
                                None,
                                placeholder="Lower"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest1_species",
                                None,
                                placeholder="Only P-glauca"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest1_site",
                                None,
                                placeholder="Alaska"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest1_amplitude",
                                None,
                                placeholder="Extreme"
                            )
                        )
                    ),

                    # Forest 2
                    ui.tags.tr(
                        ui.tags.td("2"),
                        ui.tags.td(
                            ui.input_text(
                                "forest2_temp",
                                None,
                                placeholder="Colder"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest2_humidity",
                                None,
                                placeholder="Higher"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest2_solar",
                                None,
                                placeholder="Shadier"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest2_soil",
                                None,
                                placeholder="Drier"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest2_pressure",
                                None,
                                placeholder="Higher"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest2_species",
                                None,
                                placeholder="Mixed species"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest2_site",
                                None,
                                placeholder="Mixed Sites"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest2_amplitude",
                                None,
                                placeholder="Moderate"
                            )
                        )
                    ),

                    # Forest 3
                    ui.tags.tr(
                        ui.tags.td("3"),
                        ui.tags.td(
                            ui.input_text(
                                "forest3_temp",
                                None,
                                placeholder="Warmer"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest3_humidity",
                                None,
                                placeholder="Average"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest3_solar",
                                None,
                                placeholder="Sunnier"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest3_soil",
                                None,
                                placeholder="Wetter"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest3_pressure",
                                None,
                                placeholder="Higher"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest3_species",
                                None,
                                placeholder="Mixed species"
                            )
                        ),
                        ui.tags.td(
                            ui.input_text(
                                "forest3_site",
                                None,
                                placeholder="Northwest Territories"
                            )
                        ),

                        ui.tags.td(
                            ui.input_text(
                                "forest3_amplitude",
                                None,
                                placeholder="Moderate"
                            )
                        )
                    )
                )
            )
            ),

            ui.card(
                ui.card_header("Cluster Diagnostics"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Silhouette & Cluster Sizes"),
                        ui.output_text_verbatim(
                            "kmeans_summary",
                            placeholder="Click 'Run K-Means Model' to fit clustering."
                        )
                    ),
                    ui.card(
                        ui.card_header("Change in Basal Area by Cluster"),
                        output_widget("kmeans_boxplot")
                    ),
                    col_widths=(4, 8)
                )
            )
            
        )
    ),
    


    # Tab 7: Conclusions
    ui.nav_panel(
        "Conclusions",
        ui.card(
            ui.card_header("Tree Model Conclusion"),
            ui.markdown(
                """
This project focuses on modeling the impact that environmental factors have on Tree Stem Amplitude (how much a tree grows/shrinks in a day). Tree Amplitude primarily comes from trees absorbing or losing water. As the climate changes (and the arctic is particularly susceptible to warming

It is unknown how trees will react. Our model hopes to provide clarity to what factors impact tree amplitude and therefore how climate change can impact trees.
                """
            ),
        ),
        ui.card(
            ui.card_header("Logistic Regression"),
            ui.markdown(
                """
To address the research question of distinguishing distinct physiological regimes, we utilized a multinomial logistic regression classifier to predict categories of **Basal Area Daily Amplitude**, a choice well-justified for isolating "Extreme Change" events from background noise.

The model achieved an accuracy of **47.7%**, outperforming the random baseline (33%), though performance metrics indicate a stronger ability to identify stable periods (*"No/Little Change"* Recall: 0.61) compared to detecting high-amplitude events (*"Extreme Change"* Recall: 0.33).

Despite this identification gap, the model successfully validated key biological assumptions, confirming that species identity and energy input are deterministic: *Picea mariana* and high **average solar irradiance** emerged as the strongest positive drivers of extreme daily amplitude, while *Picea glauca* served as a significant negative predictor associated with stability.
                """
            ),
        ),
        ui.card(
            ui.card_header("Principal Component Analysis and Regression"),
            ui.markdown(
                """
To condense the number of features in our analysis and potentially simplify the model, we performed principal component analysis. This technique led to the identification of 6 principal components that explain roughly 95% of the variance in the data, allowing us to shrink the number of features from 8 to 6.

While some features are more difficult to interpret (particularly the later ones), we do gain valuable insight from some of the components. For example, PC1 and PC2 appear to be related to current tree stress, with PC1 being highly correlated with current stem radius and basal area, while PC2 is negatively correlated with these factors. PC4 seems to be a good general indciator of tree growth, since it is positively correlated with change in stem radius and basal area, as well as humidity and soil water content, which may indicate that additional expansion comes from water absorption. 

When we take these components and apply them to a Principal Component Regression (PCR) model, they appear to work very well. As our scree plot demonstrated, the first four components are particularly useful in explaining the change in basal area. Overall, PCA has proven to be a useful tool.
                """
            ),
        ),

        ui.card(
            ui.card_header("Multiple Regression"),
            ui.markdown(
                """
Our regression results support the well-established hydraulic mechanism where high VPD drives reversible trunk shrinkage and recovery with low VPD. Higher VPD increased daily stem amplitude, while greater soil moisture buffered this effect, consistent with Devine & Harrington (2011).

Our results show that Arctic tree stem dynamics respond strongly to VPD and soil water, not just temperature, supporting Jensen’s argument that moisture-related stress is a key but overlooked driver of Arctic tree physiology under climate change.

This is particularly compelling as several studies have found a strong relationship between shrinkage (TWD )and hydraulic stress ($psi$) persisting across all drought conditions until lethal dehydration such as in (Ziegler et al., 2024)/ That is, large TWD (shrinkage) amplitudes are strongly linked to high hydraulic stress ($psi$ approaching lethal levels) because large TWD means living tissues have lost a lot of water (low turgor) to supply transpiration, signaling water stress.

                """
            ),
            # Add more ui cards here
        ),

        ui.card(
            ui.card_header("KNN Classification"),
            ui.markdown(
                """
To approach the question of predicting **Basal Area Daily Amplitude** from a set of environmental factors, we applied a K-Nearest Neighbors (KNN) classifier. Our model achieved an accuracy of **66%**, outperforming the random baseline (33%) by a factor of 2. The model had good predictive performance across all 3 categories, with a "Moderate Change" preforming the worst yet still having a precision of 57.7% and a recall of 52.2%.

This was found using the following features: *Air Pressure*, *Humidity*, *Temperature*, *Solar Radiation*, *Soil Moisture*, *Stem Radius*, and *Species*. The addition of variables measuring time of year (month), site location, latitude, and longitude (among others) did not improve performance. Thus, a fairly accurate predictive model can be built using only environmental features while effectively controlling in part for species in tree size. While this model cannot show exactly how influential each variable is, it does provide a useful baseline for further analysis and shows evidence that tree daily amplitude is strongly influenced by environmental factors.
                """
            ),
            # Add more ui cards here
        ),
    
        ui.card(
            ui.card_header("K-Means Clustering"),
            ui.markdown(
                """
To observe how different environmental conditions group trees into distinct "forests," we applied K-Means clustering using both numeric and categorical variables. Based on the elbow plot and nearly identical silhouette scores for three and four clusters, we chose three clusters to keep the model interpretable while still capturing key structure in the data. 

Forest 1 represents **wetter, hotter, and sunnier sites** with **extreme changes** in mean amplitude, whereas Forest 2 captures **drier, colder, and shadier environments** where mean amplitude is only **moderate**. Forest 3 is similar to Forest 1 but has **higher humidity and air pressure** that result in only **moderate** mean amplitude. This suggests that small shifts in climate variables can meaningfully change overall tree growth patterns.
                """
        ),
        # Add more ui cards here
    ),
)
)



# --- Server Logic ---
def server(input, output, session):

    # -- A. Filters & Setup --
    @reactive.Effect
    def _():
        if not df.empty:
            sites = sorted(df['site'].astype(str).unique().tolist())
            species = sorted(df['species'].astype(str).unique().tolist())
            growth = sorted(df['growth_category'].astype(str).unique().tolist())

            ui.update_selectize("filter_site", choices=sites)
            ui.update_selectize("filter_species", choices=species)
            ui.update_selectize("filter_growth", choices=growth)
            
            valid_num = [c for c in available_numeric if c in df.columns]
            valid_cat = [c for c in available_categorical if c in df.columns]
            sel_num = [c for c in default_numeric if c in valid_num]
            sel_cat = [c for c in default_categorical if c in valid_cat]

            ui.update_selectize("model_numeric", choices=valid_num, selected=sel_num)
            ui.update_selectize("model_categorical", choices=valid_cat, selected=sel_cat)
            
            if growth:
                ui.update_radio_buttons("target_view", choices=growth, selected=growth[0])

    @reactive.Effect
    @reactive.event(input.reset_filters)
    def _():
        ui.update_slider("date_range", value=[min_date, max_date])
        ui.update_selectize("filter_site", selected=[])
        ui.update_selectize("filter_species", selected=[])
        ui.update_selectize("filter_growth", selected=[])

    @reactive.Calc
    def filtered_df():
        if df.empty: return pd.DataFrame()
        out = df.copy()
        start_date, end_date = input.date_range()
        out = out[(out['Date'] >= start_date) & (out['Date'] <= end_date)]
        if input.filter_site(): out = out[out['site'].astype(str).isin(input.filter_site())]
        if input.filter_species(): out = out[out['species'].astype(str).isin(input.filter_species())]
        if input.filter_growth(): out = out[out['growth_category'].astype(str).isin(input.filter_growth())]
        return out

    # -- B. Logistic Regression Logic --
    @reactive.Calc
    def model_results():
        if df.empty: return {}
        
        target = 'growth_category'
        numeric_features = list(input.model_numeric())
        categorical_features = list(input.model_categorical())
        
        if not numeric_features and not categorical_features:
            return None

        # Check existing columns
        existing_numeric = [c for c in numeric_features if c in df.columns]
        existing_cat = [c for c in categorical_features if c in df.columns]

        X = df[existing_numeric + existing_cat]
        y = df[target]

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, existing_numeric),
            ('cat', categorical_transformer, existing_cat)
        ])
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        return {
            "model": model, "X_test": X_test, "y_test": y_test, "score": score,
            "report_txt": classification_report(y_test, y_pred),
            "report_dict": classification_report(y_test, y_pred, output_dict=True),
            "numeric_features": existing_numeric, "categorical_features": existing_cat
        }

    # -- C. PCA Logic --
    @reactive.Calc
    def pca_results():
        if df.empty: return None
        
        potential_cols = available_numeric
        pca_cols = [c for c in potential_cols
            if (
            c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])
            and c != 'change_basal_area'
            )]
        
        if not pca_cols: return None

        X_pca = df[pca_cols].dropna()
        
        pca_pipe = Pipeline(steps=[("scaler", StandardScaler()), ("pca", PCA())])
        pca_pipe.fit(X_pca)
        pca_model = pca_pipe.named_steps["pca"]
        
        explained_var = pca_model.explained_variance_ratio_
        ev_df = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(len(explained_var))],
            "ExplainedVariance": explained_var
        })

        X_pca_scores = pca_pipe.transform(X_pca)
        scores_df = pd.DataFrame(
            X_pca_scores, 
            columns=[f"PC{i+1}" for i in range(X_pca_scores.shape[1])], 
            index=X_pca.index
        )
        
        if 'species' in df.columns: scores_df['species'] = df.loc[X_pca.index, 'species']
        if 'site' in df.columns: scores_df['site'] = df.loc[X_pca.index, 'site']
        if 'growth_category' in df.columns: scores_df['Growth'] = df.loc[X_pca.index, 'growth_category']
        
        # Create alldata_w_pcs with only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        alldata_w_pcs = df[numeric_cols].copy()
        
        # Add PC columns at the beginning, matching by index
        pc_cols = [c for c in scores_df.columns if c.startswith('PC')]
        for col in pc_cols:
            alldata_w_pcs.loc[X_pca.index, col] = scores_df[col]
        
        # Reorder columns to put PCs first
        pc_columns_present = [c for c in pc_cols if c in alldata_w_pcs.columns]
        other_columns = [c for c in alldata_w_pcs.columns if c not in pc_columns_present]
        alldata_w_pcs = alldata_w_pcs[pc_columns_present + other_columns]
        
        # Calculate loadings
        n_components = pca_model.components_.shape[0]
        loadings = pca_model.components_.T
        loading_df = pd.DataFrame(
            loadings,
            index=pca_cols,
            columns=[f"PC{i+1}_loading" for i in range(n_components)]
        )
        
        return {
            "X_pca": X_pca, 
            "ev_df": ev_df, 
            "scores_df": scores_df, 
            "corr_matrix": X_pca.corr(),
            "alldata_w_pcs": alldata_w_pcs,
            "loading_df": loading_df,
            "pca_cols": pca_cols
        }

        
    # -- D. KNN Logic --
    @reactive.Calc
    @reactive.event(input.run_knn)
    def knn_results():
        if df.empty: return {}

        # vars and train/test
        target_col = 'growth_category'
        cat_cols = list(input.knn_model_cat())
        num_cols = list(input.knn_model_num())
        num_cols = [c for c in num_cols if c in df.columns]
        cat_cols = [c for c in cat_cols if c in df.columns]
        labels = list(df[target_col].unique())

        y = df[target_col]
        X = df[cat_cols + num_cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # set up pipeline
        preprocess = ColumnTransformer(
            transformers=[
                ('num - scaler', StandardScaler(), num_cols),
                ('cat - ordinal', OrdinalEncoder(), cat_cols)
            ]
        )

        pipe = Pipeline([
            ('preprocessor', preprocess),
            ('knn', KNeighborsClassifier(weights='distance'))
        ])

        # run grid search
        param_grid = {'knn__n_neighbors': range(1, 100, 2)}

        grid = GridSearchCV(pipe, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        # get best K
        k_results_df = pd.DataFrame(grid.cv_results_)

        k_results_df['k'] = k_results_df['param_knn__n_neighbors']
        k_results_df['mean_score'] = k_results_df['mean_test_score']

        best_k = grid.best_params_['knn__n_neighbors']
        best_score = grid.best_score_

        # final model
        knn_final = Pipeline([
            ('preprocessor', preprocess),
            ('knn', KNeighborsClassifier(n_neighbors=best_k, weights='distance'))
        ])

        knn_final.fit(X_train, y_train)
        y_pred = knn_final.predict(X_test)

        # results        
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        perf_data = {
            'Class': labels,
            'Precision': [report_dict[c]['precision'] for c in labels],
            'Recall':    [report_dict[c]['recall'] for c in labels]
        }
        df_perf = pd.DataFrame(perf_data)
        df_perf_melted = df_perf.melt(id_vars='Class', var_name='Metric', value_name='Score')

        knn_accuracy = report_dict['accuracy']
        knn_b_accuracy = balanced_accuracy_score(y_test, y_pred)

        # Prepare PCA data
        y_true = y_test.copy().reset_index(drop=True)
        y_true.columns = ['target']
        y_pred = pd.DataFrame(y_pred, columns=['pred'])
        
        # Compute PCA
        pca = PCA(n_components=2)
        x_num = X_test[num_cols]
        X_pca = pd.DataFrame(pca.fit_transform(x_num), columns=["PC1", "PC2"])

        # Combine for plotting
        df_plot = pd.concat([X_pca, y_true, y_pred], axis=1)
        df_plot.rename(columns={'growth_category': 'target'}, inplace=True)
        df_plot["correct"] = df_plot["target"] == df_plot["pred"]
        df_plot["correct"] = df_plot["correct"].map({True: "Correct", False: "Incorrect"})

        return {
            "knn_accuracy_score": knn_accuracy,
            "knn_b_accuracy_score": knn_b_accuracy,
            "labels": labels,
            "x_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "k_results_df": k_results_df,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            # "model": knn_final,
            # "report_dict": report_dict,
            "best_k": best_k,
            "best_score": best_score,
            # "perf_data": perf_data,
            # "df_perf": df_perf,
            "df_perf_melted": df_perf_melted,
            "df_plot": df_plot
        }
    
    # -- E. K-Means Logic --
    @reactive.Calc
    @reactive.event(input.run_kmeans)
    def kmeans_results():
        if df.empty:
            return {}

        num_cols = list(input.kmeans_model_num())
        cat_cols = list(input.kmeans_model_cat())

        existing_num = [c for c in num_cols if c in df.columns]
        existing_cat = [c for c in cat_cols if c in df.columns]

        if not existing_num and not existing_cat:
            return {}

        X = df[existing_num + existing_cat].dropna()
        if X.empty:
            return {}

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), existing_num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), existing_cat)
            ]
        )

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("kmeans", KMeans(init="k-means++", n_init=10, random_state=6021))
            ]
        )

        k_values = list(range(2, 11))
        wcss = []

        for k in k_values:
            pipe.set_params(kmeans__n_clusters=k)
            pipe.fit(X)
            inertia = pipe["kmeans"].inertia_
            wcss.append(inertia)

        chosen_k = int(input.kmeans_k())
        if chosen_k < min(k_values) or chosen_k > max(k_values):
            chosen_k = 3  # safe fallback

        pipe.set_params(kmeans__n_clusters=chosen_k)
        pipe.fit(X)

        labels = pipe["kmeans"].labels_

        df_clusters = df.loc[X.index].copy()
        df_clusters["cluster"] = labels

        try:
            X_transformed = pipe["preprocessor"].transform(X)
            sil = float(silhouette_score(X_transformed, labels))
        except Exception:
            sil = None

        cluster_sizes = df_clusters["cluster"].value_counts().sort_index()

        return {
            "chosen_k": chosen_k,
            "wcss_k_values": k_values,
            "wcss": wcss,
            "silhouette": sil,
            "cluster_sizes": cluster_sizes,
            "df_clusters": df_clusters,
            "numeric_cols": existing_num,
            "categorical_cols": existing_cat,
        }


    # -- Outputs --
    @render.data_frame
    def data_table():
        return render.DataTable(filtered_df())

    @render.text
    def accuracy_score():
        res = model_results()
        if not res: return "Please select at least one variable."
        return f"{res['score']:.4f}"

    @render.text
    def class_report():
        res = model_results()
        if not res: return ""
        return res['report_txt']

    @render.plot
    def analysis_plot():
        res = model_results()
        if not res: return
        
        model = res["model"]
        report_dict = res["report_dict"]
        num_feats = res["numeric_features"]
        cat_feats = res["categorical_features"]
        
        cat_names = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_feats)
        feature_names = num_feats + list(cat_names)
        
        target_class = input.target_view()
        coefficients = []

        if hasattr(model.named_steps['classifier'], 'coef_'):
            coefs = model.named_steps['classifier'].coef_
            
            if coefs.shape[0] == 1:
                positive_class = model.classes_[1]
                raw_coefs = coefs[0]
                if positive_class == target_class:
                    coefficients = raw_coefs
                else:
                    coefficients = -raw_coefs
            else:
                if target_class in model.classes_:
                    class_index = list(model.classes_).index(target_class)
                    coefficients = coefs[class_index]
        
        if len(coefficients) > 0:
            df_coef = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
            df_coef = df_coef.reindex(df_coef.Coefficient.abs().sort_values(ascending=False).index).head(10)
        else:
            df_coef = pd.DataFrame({'Feature': ['N/A'], 'Coefficient': [0]})

        # Performance Metrics
        classes_to_plot = [c for c in model.classes_ if c in report_dict and c != 'accuracy']
        perf_data = {
            'Class': classes_to_plot,
            'Precision': [report_dict[c]['precision'] for c in classes_to_plot],
            'Recall':    [report_dict[c]['recall'] for c in classes_to_plot]
        }
        df_perf = pd.DataFrame(perf_data).melt(id_vars='Class', var_name='Metric', value_name='Score')

        # --- Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.set_style("whitegrid")
        
        # 1. Feature Importance (Dynamic Colors)
        colors = ['#2ca02c' if c > 0 else '#d62728' for c in df_coef['Coefficient']]
        
        sns.barplot(
            x='Coefficient', 
            y='Feature', 
            data=df_coef, 
            ax=axes[0], 
            palette=colors, 
            edgecolor='black'
        )
        axes[0].set_title(f"Top 10 Drivers of '{target_class}'", fontsize=14, weight='bold')
        axes[0].set_xlabel("Impact (Log-Odds Coefficient)")
        axes[0].axvline(0, color='black', linewidth=1)
        
        min_c, max_c = df_coef['Coefficient'].min(), df_coef['Coefficient'].max()
        padding = max(0.3, (max_c - min_c) * 0.2)
        axes[0].set_xlim(min_c - padding, max_c + padding)
        
        for i, v in enumerate(df_coef['Coefficient']):
            axes[0].text(v + (padding * 0.05 if v > 0 else -padding * 0.05), i, f"{v:+.2f}", va='center', ha='left' if v > 0 else 'right', fontsize=10, color='black')

        # 2. Performance (Random Chance = 0.50)
        if not df_perf.empty:
            sns.barplot(x='Class', y='Score', hue='Metric', data=df_perf, ax=axes[1], palette='RdBu', edgecolor='black')
            axes[1].set_title("Model Performance by Category", fontsize=14, weight='bold')
            axes[1].set_ylim(0, 1.0)
            axes[1].axhline(0.33, color='red', linestyle='--', label='Random Chance (0.33)')
            axes[1].legend(loc='lower center')
        
        plt.tight_layout()
        return fig

    # -- Outputs: PCA --
    @render_widget
    def corr_matrix_plot():
        res = pca_results()
        if not res: return go.Figure()
        return px.imshow(res["corr_matrix"], text_auto=True, color_continuous_scale="RdBu_r", aspect="auto", title="Correlation Matrix")

    @render_widget
    def scree_plot():
        res = pca_results()
        if not res: return go.Figure()
        return px.line(res["ev_df"], x="PC", y="ExplainedVariance", markers=True, title="Scree Plot")

    @render_widget
    def pca_scatter_plot():
        res = pca_results()
        if not res: return go.Figure()
        
        sdf = res["scores_df"]
        
        # Get user-selected PCs
        x_col = input.pca_x_axis()
        y_col = input.pca_y_axis()
        
        # Check if selected PCs exist in the data
        if x_col not in sdf.columns:
            x_col = "PC1"
        if y_col not in sdf.columns:
            y_col = "PC2"
        
        # Determine color column based on user selection
        color_choice = input.pca_color()
        if color_choice == "Growth" and "Growth" in sdf.columns:
            color_col = "Growth"
        elif color_choice == "Species" and "species" in sdf.columns:
            color_col = "species"
        elif color_choice == "Site" and "site" in sdf.columns:
            color_col = "site"
        else:
            # Fallback
            color_col = "Growth" if "Growth" in sdf.columns else ("species" if "species" in sdf.columns else None)
        
        # Prepare hover data - only include columns that exist
        hover_cols = []
        if "species" in sdf.columns:
            hover_cols.append("species")
        if "site" in sdf.columns:
            hover_cols.append("site")
        if "Growth" in sdf.columns:
            hover_cols.append("Growth")
        
        # Create the scatter plot with marginal distributions
        fig = px.scatter(
            sdf,
            x=x_col,
            y=y_col,
            color=color_col,
            marginal_x="histogram",
            marginal_y="histogram",
            title=f"PCA Biplot with Marginal Distributions ({x_col} vs {y_col})",
            hover_data=hover_cols if hover_cols else None,
            opacity=0.7
        )
        
        fig.update_layout(width=900, height=800)
        
        return fig

    @render_widget
    def corr_matrix_with_pcs_plot():
        res = pca_results()
        if not res: return go.Figure()
        
        # Get the dataframe with PCs included
        alldata_w_pcs = res["alldata_w_pcs"]
        
        # Calculate correlation matrix (dropna to handle missing values)
        corr_matrix_with_pcs = alldata_w_pcs.corr()
        
        return px.imshow(
            corr_matrix_with_pcs, 
            text_auto=True, 
            color_continuous_scale="RdBu_r", 
            aspect="auto", 
            title="Correlation Matrix (Including Principal Components)",
            height=800  # Make it taller since there are more variables
        )

    @render_widget
    def loadings_plot():
        res = pca_results()
        if not res: return go.Figure()
        
        loading_df = res["loading_df"]
        scores_df = res["scores_df"]
        
        # Get user-selected PCs
        x_pc = input.loadings_x_axis()
        y_pc = input.loadings_y_axis()
        
        x_loading_col = f"{x_pc}_loading"
        y_loading_col = f"{y_pc}_loading"
        
        # Check if selected loadings exist
        if x_loading_col not in loading_df.columns:
            x_loading_col = "PC1_loading"
            x_pc = "PC1"
        if y_loading_col not in loading_df.columns:
            y_loading_col = "PC2_loading"
            y_pc = "PC2"
        
        arrow_scale = input.arrow_scale()
        
        fig = go.Figure()
        
        # Add scatter points for observations
        if x_pc in scores_df.columns and y_pc in scores_df.columns:
            hover_text = []
            for idx in scores_df.index:
                parts = []
                if 'species' in scores_df.columns:
                    parts.append(f"Species: {scores_df.loc[idx, 'species']}")
                if 'site' in scores_df.columns:
                    parts.append(f"Site: {scores_df.loc[idx, 'site']}")
                if 'Growth' in scores_df.columns:
                    parts.append(f"Growth: {scores_df.loc[idx, 'Growth']}")
                hover_text.append("<br>".join(parts))
            
            fig.add_trace(go.Scatter(
                x=scores_df[x_pc],
                y=scores_df[y_pc],
                mode="markers",
                marker=dict(size=5, opacity=0.5, color='lightblue'),
                name="Observations",
                hovertext=hover_text,
                hoverinfo="text"
            ))
        
        # Add loading arrows
        for var_name, row in loading_df.iterrows():
            x_arrow = row[x_loading_col] * arrow_scale
            y_arrow = row[y_loading_col] * arrow_scale
            
            fig.add_trace(go.Scatter(
                x=[0, x_arrow],
                y=[0, y_arrow],
                mode="lines+markers+text",
                text=[None, var_name],
                textposition="top center",
                line=dict(color='red', width=2),
                marker=dict(size=8, color='red'),
                showlegend=False,
                hoverinfo='text',
                hovertext=f"{var_name}<br>{x_pc}: {row[x_loading_col]:.3f}<br>{y_pc}: {row[y_loading_col]:.3f}"
            ))
        
        fig.update_layout(
            title=f"PCA Loadings Biplot: {x_pc} vs {y_pc}",
            xaxis_title=f"{x_pc} Loadings",
            yaxis_title=f"{y_pc} Loadings",
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            width=900,
            height=800,
            hovermode='closest'
        )
        
        return fig
        
        # Principal Component Regression Section
    @reactive.Calc
    def pcr_results():
        res = pca_results()
        if not res:
            return None
        
        alldata_w_pcs = res["alldata_w_pcs"]
        
        # Check if target variable exists
        if 'change_basal_area' not in df.columns:
            return None
        
        # Get indices where we have both PCA scores and target variable
        valid_indices = alldata_w_pcs.index.intersection(df.index)
        
        # Prepare features (all PC columns) - excluding change_basal_area
        pc_cols = [c for c in alldata_w_pcs.columns if c.startswith('PC')]
        if not pc_cols:
            return None
        
        X = alldata_w_pcs.loc[valid_indices, pc_cols].dropna()
        y = df.loc[X.index, 'change_basal_area'].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 10:  # Need minimum samples
            return None
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Try different numbers of components
        results = []
        for n_pcs in range(1, min(len(pc_cols) + 1, 10)):
            pcr_pipe = Pipeline([
                ("pca", PCA(n_components=n_pcs)),
                ("linreg", LinearRegression())
            ])
            pcr_pipe.fit(X_train, y_train)
            
            train_r2 = pcr_pipe.score(X_train, y_train)
            test_r2 = pcr_pipe.score(X_test, y_test)
            
            y_train_pred = pcr_pipe.predict(X_train)
            y_test_pred = pcr_pipe.predict(X_test)
            
            train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
            test_rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))
            
            results.append({
                'n_components': n_pcs,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            })
        
        results_df = pd.DataFrame(results)
        
        # Fit model with optimal number of components (based on test R²)
        best_n = results_df.loc[results_df['test_r2'].idxmax(), 'n_components']
        
        pcr_pipe_best = Pipeline([
            ("pca", PCA(n_components=int(best_n))),
            ("linreg", LinearRegression())
        ])
        pcr_pipe_best.fit(X_train, y_train)
        
        y_test_pred_best = pcr_pipe_best.predict(X_test)
        
        return {
            'results_df': results_df,
            'best_n': int(best_n),
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred_best,
            'best_model': pcr_pipe_best,
            'X_train': X_train,
            'y_train': y_train
        }

    # Add these outputs to your server function

    # PCR with selected components
    @reactive.Calc
    @reactive.event(input.update_pcr)
    def pcr_selected_results():
        res = pca_results()
        if not res:
            return None
        
        alldata_w_pcs = res["alldata_w_pcs"]
        
        # Check if target variable exists
        if 'change_basal_area' not in df.columns:
            return None
        
        # Get selected PCs from checkbox
        selected_pcs = list(input.pcr_selected_pcs())
        if not selected_pcs:
            return None
        
        # Get indices where we have both PCA scores and target variable
        valid_indices = alldata_w_pcs.index.intersection(df.index)
        
        # Filter to only selected PCs
        available_pcs = [c for c in selected_pcs if c in alldata_w_pcs.columns]
        if not available_pcs:
            return None
        
        X = alldata_w_pcs.loc[valid_indices, available_pcs].dropna()
        y = df.loc[X.index, 'change_basal_area'].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 10:  # Need minimum samples
            return None
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Fit linear regression model (no PCA transform needed, already have PCs)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
        test_rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))
        
        return {
            'selected_pcs': selected_pcs,
            'n_components': len(available_pcs),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'model': model
        }
    
    @render.text
    def pcr_metrics():
        res = pcr_results()
        if not res:
            return "PCR analysis not available. Ensure 'change_basal_area' exists in your data."
        
        best_n = res['best_n']
        results_df = res['results_df']
        best_row = results_df[results_df['n_components'] == best_n].iloc[0]
        
        output = f"Principal Component Regression Results\n"
        output += f"=" * 50 + "\n\n"
        output += f"Optimal number of components: {best_n}\n\n"
        output += f"Train R²  : {best_row['train_r2']:.4f}\n"
        output += f"Test  R²  : {best_row['test_r2']:.4f}\n"
        output += f"Train RMSE: {best_row['train_rmse']:.4f}\n"
        output += f"Test  RMSE: {best_row['test_rmse']:.4f}\n"
        
        return output

    @render_widget
    def pcr_pred_vs_actual():
        # Try to get selected results first, fall back to automatic results
        res = pcr_selected_results()
        if not res:
            res = pcr_results()
        if not res:
            return go.Figure()
        
        y_test = res['y_test']
        y_test_pred = res['y_test_pred']
        n_comp = res['n_components']
        
        # Get selected PCs if available
        if 'selected_pcs' in res:
            pc_info = f"Selected PCs: {', '.join(res['selected_pcs'])}"
        else:
            pc_info = f"n_components={n_comp}"
        
        fig = px.scatter(
            x=y_test, 
            y=y_test_pred,
            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
            title=f'PCR: Predicted vs Actual (Test Set, {pc_info})',
            opacity=0.6
        )
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        
        fig.add_scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
        
        fig.update_layout(width=900, height=600)
        
        return fig

    @render_widget
    def pcr_metrics_plot():
        res = pcr_results()
        if not res:
            return go.Figure()
        
        results_df = res['results_df']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results_df['n_components'],
            y=results_df['test_r2'],
            mode='lines+markers',
            name='Test R²',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=results_df['n_components'],
            y=results_df['train_r2'],
            mode='lines+markers',
            name='Train R²',
            line=dict(color='lightblue', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='PCR Performance vs Number of Components',
            xaxis_title='Number of Principal Components',
            yaxis_title='R² Score',
            width=700,
            height=500
        )
        
        return fig

    # Outputs Multiple Regression
    @render.text
    @reactive.event(input.run_mixedlm)  # only runs when button clicked
    def mixedlm_output():
        import statsmodels.formula.api as smf
        
        # Filter for P glauca
        df_filtered = df[df["species"] == "P glauca"]
        
        # Fit the mixed effects model
        model = smf.mixedlm(
            "change_stem_radius ~ VPD * avg_soil_water_content + avg_solar_irradiance",
            data=df_filtered,
            groups=df_filtered["tree"]
        ).fit()
    
    # Return the summary as text
        return str(model.summary())
    
    @render.text
    @reactive.event(input.run_ols)  # only runs when button clicked
    def ols_output():
        import statsmodels.formula.api as smf
        
        # Filter for P glauca
        df_filtered = df[df["species"] == "P glauca"]
        
        # Fit the mixed effects model
        model = smf.ols(
            formula="""
                change_stem_radius ~ VPD * avg_soil_water_content + avg_solar_irradiance + C(tree)
            """,
            data=df_filtered
        ).fit()
    
    # Return the summary as text
        return str(model.summary())
    

    # -- Output KNN --
    @render.text
    def knn_accuracy_score():
        res = knn_results()
        if not res: return "Accuracy: "
        return f"Accuracy: {res['knn_accuracy_score']:.4f}"
    
    @render.text
    def knn_b_accuracy_score():
        res = knn_results()
        if not res: return "Balanced Accuracy: "
        return f"Balanced Accuracy: {res['knn_b_accuracy_score']:.4f}"
    
    @render_widget
    def knn_k_selection_plot():
        res = knn_results()
        if not res: return go.Figure()

        k_results_df = res['k_results_df']
        best_k = res['best_k']
        best_score = res['best_score']

        fig = go.Figure()

        # Main line
        fig.add_trace(go.Scatter(
            x=k_results_df['k'],
            y=k_results_df['mean_score'],
            mode='lines+markers',
            name='Mean CV Balanced Accuracy',
            hovertemplate="k=%{x}<br>Score=%{y:.4f}<extra></extra>"
        ))

        # Best k marker
        fig.add_trace(go.Scatter(
            x=[best_k],
            y=[best_score],
            mode='markers+text',
            text=[f'Best k = {best_k}'],
            textposition='top center',
            marker=dict(size=10),
            name='Best k',
            hovertemplate="Best k=%{x}<br>Score=%{y:.4f}<extra></extra>"
        ))

        # Layout
        fig.update_layout(
            title=f"Cross-Validated Balanced Accuracy vs. K (best k = {best_k})",
            xaxis_title="Number of Neighbors (k)",
            yaxis_title="Mean CV Balanced Accuracy",
            hovermode="x unified",
            template="plotly_white",
            yaxis=dict(title="Mean CV Balanced Accuracy"),
            xaxis=dict(title="Number of Neighbors (k)")
        )

        return fig
    
    @render_widget
    def knn_confusion_matrix():
        res = knn_results()
        if not res: return go.Figure()

        y_test = res['y_test']
        y_pred = res['y_pred']
        labels = res['labels']
        labels_2 = [str(i).split(" ")[0] for i in labels]

        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # Text annotations
        text = [[str(cm[i][j]) for j in range(len(labels))] for i in range(len(labels))]

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels_2,
            y=labels_2,
            colorscale="Blues",
            showscale=True,
            text=text,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))

        fig.update_layout(
            title="Confusion Matrix",
            xaxis=dict(title="Predicted Label"),
            yaxis=dict(title="Actual Label", autorange="reversed"),
            template="plotly_white"
        )

        return fig
   
    @render_widget
    def knn_performance_plot():
        res = knn_results()
        if not res: return go.Figure()

        df_perf_melted = res['df_perf_melted']

        classes = df_perf_melted['Class'].unique()
        classes_print = [str(i).split(" ")[0] for i in classes]
        metrics = df_perf_melted['Metric'].unique()

        # Optional color palette similar to seaborn's RdBu
        colors = ['#67a9cf', '#ef8a62', '#2166ac', '#b2182b']

        fig = go.Figure()

        # Add a trace for each Metric (grouped bars)
        for i, metric in enumerate(metrics):
            df_metric = df_perf_melted[df_perf_melted['Metric'] == metric]

            fig.add_trace(go.Bar(
                x=df_metric['Class'],
                y=df_metric['Score'],
                name=metric,
                marker=dict(
                    color=colors[i % len(colors)],
                    line=dict(color='black', width=1)  # black edges like seaborn
                )
            ))

        # Add horizontal reference line at 0.33
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(classes)-0.5,
            y0=0.33,
            y1=0.33,
            line=dict(color="black", width=2, dash="dash"),
        )

        fig.add_annotation(
            x=len(classes)/2,
            y=0.35,
            text="Random Chance (0.33)",
            showarrow=False,
            font=dict(color="black")
        )

        fig.update_layout(
            barmode='group',
            title="Per-Class Performance Metrics",
            xaxis_title="Class",
            yaxis_title="Score",
            template="plotly_white",
            legend_title="Metric",
        )

        return fig
    
    @render_widget
    def knn_scatter_plot():
        res = knn_results()
        if not res:
            return go.Figure()

        # Get the dropdown selection
        knn_scatter_color_by = input.knn_scatter_color_by()  # 'target', 'pred', 'correct'

        y_test = res['y_test'].copy()
        y_pred = res['y_pred'].copy()
        pca_real = pca_results()['alldata_w_pcs'][['PC1', 'PC2']].copy()

        df_plot_old = res['df_plot']
        
        y_test_2 = y_test.copy().reset_index()
        y_pred_2 = pd.DataFrame(y_pred, columns=['pred'])
        y_combined = pd.concat([y_test_2, y_pred_2], axis=1)
        y_combined.index = y_combined['index']
        df_plot = pd.concat([pca_real, y_combined], axis=1, join='inner')
        df_plot = df_plot.reset_index(drop=True)
        df_plot.rename(columns={'growth_category': 'target'}, inplace=True)
        df_plot = df_plot[['PC1', 'PC2', 'target', 'pred']]
        df_plot["correct"] = df_plot["target"] == df_plot["pred"]
        df_plot["correct"] = df_plot["correct"].map({True: "Correct", False: "Incorrect"})


        # df_plot = pd.concat([pca_real, y_test], axis=1, join="inner")
        # y_pred = pd.Series(y_pred['pred'], index=y_test.index, name="pred")
        # y_pred = y_pred.loc[df_plot.index] 
        # df_plot["y_pred"] = y_pred
        # df_plot = df_plot.reset_index(drop=True)
        # test = df_plot.isna().sum()



        # Create scatter plot
        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color=df_plot[knn_scatter_color_by],
            hover_data=df_plot.columns,
            title="PCA Scatter Plot",
            opacity=0.65
        )


        # Update legend title nicely
        legend_title_map = {
            "target": "True Class",
            "pred": "Predicted Class",
            "correct": "Correct / Incorrect"
        }
        fig.update_layout(legend_title_text=legend_title_map.get(knn_scatter_color_by, knn_scatter_color_by))

        return fig

    @render_widget
    def kmeans_elbow_plot():
        res = kmeans_results()
        if not res:
            return go.Figure()

        k_values = res["wcss_k_values"]
        wcss = res["wcss"]
        chosen_k = res["chosen_k"]

        fig = go.Figure()

        # Main WCSS line
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=wcss,
                mode="lines+markers",
                name="WCSS / Inertia",
                hovertemplate="k=%{x}<br>WCSS=%{y:.2f}<extra></extra>"
            )
        )

        # Highlight chosen k
        if chosen_k in k_values:
            idx = k_values.index(chosen_k)
            fig.add_trace(
                go.Scatter(
                    x=[chosen_k],
                    y=[wcss[idx]],
                    mode="markers+text",
                    text=[f"k = {chosen_k}"],
                    textposition="top center",
                    marker=dict(size=10),
                    name="Chosen k"
                )
            )

        fig.update_layout(
            title="Elbow Plot: WCSS vs Number of Clusters (k)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="WCSS / Inertia",
            template="plotly_white",
            hovermode="x unified"
        )

        return fig
    
    
    @render.text
    def kmeans_summary():
        res = kmeans_results()
        if not res:
            return "Click 'Run K-Means Model' to fit clustering."

        lines = []
        k = res["chosen_k"]
        sil = res["silhouette"]
        sizes = res["cluster_sizes"]

        if sil is not None:
            lines.append(f"Silhouette score for k = {k}: {sil:.3f}")
        else:
            lines.append(f"Silhouette score for k = {k}: not available")

        lines.append("")
        lines.append("Cluster sizes:")
        for cluster_id, n in sizes.items():
            lines.append(f"  Cluster {cluster_id}: {n} observations")

        return "\n".join(lines)

    @render_widget
    def kmeans_boxplot():
        res = kmeans_results()
        if not res:
            return go.Figure()

        df_clusters = res["df_clusters"]
        if "change_basal_area" not in df_clusters.columns:
            return go.Figure()

        fig = go.Figure()

        for cluster_id, group in df_clusters.groupby("cluster"):
            fig.add_trace(
                go.Box(
                    y=group["change_basal_area"],
                    name=f"Cluster {cluster_id}",
                    boxpoints="outliers"
                )
            )

        fig.update_layout(
            title="Change in Basal Area by Cluster",
            yaxis_title="change_basal_area",
            xaxis_title="Cluster",
            template="plotly_white"
        )

        return fig
        
    @render.table
    def kmeans_forest_table():
        # Simple table linking cluster IDs to your forest descriptions
        df_forest = pd.DataFrame({
            "Forest": ["Forest 1", "Forest 2", "Forest 3"],
            "Cluster ID": [0, 1, 2],
            "Description": [
                input.forest1_desc(),
                input.forest2_desc(),
                input.forest3_desc()
            ]
        })
        return df_forest
    
    @render.plot
    def kmeans_temp_humidity():
        res = kmeans_results()
        if not res:
            return

        df_clusters = res["df_clusters"].copy()
        df_clusters["forest"] = df_clusters["cluster"] + 1

        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            data=df_clusters,
            x="average_air_temperature",
            y="average_humidity",
            hue="forest"
        )
        plt.title("Temperature vs Humidity by Forest")
        plt.xlabel("Average Air Temperature")
        plt.ylabel("Average Humidity")
        plt.legend(title="Forest")
        plt.tight_layout(pad=2)

    @render.plot
    def kmeans_solar_soil():
        res = kmeans_results()
        if not res:
            return

        df_clusters = res["df_clusters"].copy()
        df_clusters["forest"] = df_clusters["cluster"] + 1

        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            data=df_clusters,
            x="avg_solar_irradiance",
            y="avg_soil_water_content",
            hue="forest"
        )
        plt.title("Solar Irradiance vs Soil Water Content by Forest")
        plt.xlabel("Average Solar Irradiance")
        plt.ylabel("Average Soil Water Content")
        plt.legend(title="Forest")
        plt.tight_layout(pad=2)
        

    @render.plot
    def kmeans_pressure_humidity():
        res = kmeans_results()
        if not res:
            return

        df_clusters = res["df_clusters"].copy()
        df_clusters["forest"] = df_clusters["cluster"] + 1

        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            data=df_clusters,
            x="average_air_pressure",
            y="average_humidity",
            hue="forest"
        )
        plt.title("Air Pressure vs Humidity by Forest")
        plt.xlabel("Average Air Pressure")
        plt.ylabel("Average Humidity")
        plt.legend(title="Forest")
        plt.tight_layout(pad=2)

    @render.plot
    def kmeans_temp_basal():
        res = kmeans_results()
        if not res:
            return

        df_clusters = res["df_clusters"].copy()
        df_clusters["forest"] = df_clusters["cluster"] + 1

        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            data=df_clusters,
            x="average_air_temperature",
            y="change_basal_area",
            hue="forest"
        )

        overall_mean = df_clusters["change_basal_area"].mean()
        plt.axhline(y=overall_mean, linestyle="--", color="black")
        xmin, xmax = plt.xlim()
        plt.text(
            xmax,
            overall_mean,
            "Overall average",
            ha="right",
            va="bottom"
        )

        plt.ylim(0, 2)
        plt.title("Temperature vs Amplitude by Forest")
        plt.xlabel("Average Air Temperature")
        plt.ylabel("Amplitude")
        plt.legend(title="Forest")
        plt.tight_layout(pad=2)

    
app = App(app_ui, server)
