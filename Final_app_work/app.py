import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget  # Required for Plotly
from datetime import date
import plotly.graph_objects as go

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

# Defaults
default_numeric = ['average_humidity', 'avg_solar_irradiance', 'avg_soil_water_content', 'average_air_pressure','average_stem_radius']
default_categorical = ['freeze_flag', 'species', 'site', 'plot']


# --- UI Definition ---
app_ui = ui.page_navbar(
    # Tab 1: Readme
    ui.nav_panel(
        "Readme",
        ui.markdown(
            """
            # Project Overview: Plant Growth Prediction
            
            This application allows researchers to interactively analyze plant growth outcomes using Logistic Regression and Principal Component Analysis (PCA).
            
            ### 1. The Goal
            Our primary objective is to determine which environmental factors—such as soil moisture or solar intensity—are the strongest predictors of robust plant growth.

            ### 2. Methodologies
            * **Multiple Regression**: A statistical modeling approach used to quantify how environmental variables influence stem radius amplitude, the size of the daily swelling–shrinking cycle of the tree stem. By modeling predictors like VPD, soil moisture, and solar exposure (and their interactions), the MLR helps identify which conditions most strongly increase or dampen these daily changes in stem size.
            * **Logistic Regression**: A predictive model used to classify growth. You can now customize the inputs and explore drivers for specific growth categories (e.g., "A Little Growth" vs "A Lot of Growth").
            * **Principal Component Analysis (PCA)**: A dimensionality reduction technique used to explore the structure of the data and identify natural clusters.
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
                    ui.p("Use the sidebar on the left to add or remove variables from the model. You can also toggle the graph to see predictors for 'A Little Growth' vs 'A Lot of Growth'."),

                    ui.h5("What is Logistic Regression?"),
                    ui.p("Unlike linear regression, which predicts a number, Logistic Regression predicts the probability of an event happening (like a plant growing a lot)."),
                    
                    ui.h5("Understanding the Outputs"),
                    ui.tags.ul(
                        ui.tags.li(ui.strong("Precision"), ": When the model predicts 'A Lot of Growth', how often is it correct? (Quality)."),
                        ui.tags.li(ui.strong("Recall"), ": Out of all actual 'A Lot of Growth' cases, how many did the model find? (Quantity)."),
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
                ui.card_header("1. Correlation Matrix (Original Variables)"),
                output_widget("corr_matrix_plot")
            ),
            ui.card(
                ui.card_header("2. Scree Plot (Variance Explained)"),
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
        # Optional: Add correlation matrix with PCs
        ui.layout_columns(
            ui.card(
                ui.card_header("4. Correlation Matrix (All Variables + PCs)"),
                output_widget("corr_matrix_with_pcs_plot")
            ),
            col_widths=(12,)
        )
    ),
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
        pca_cols = [c for c in potential_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        
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
        # Get all numeric columns from original dataframe
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Start with numeric data
        alldata_w_pcs = df[numeric_cols].copy()
        
        # Add PC columns at the beginning, matching by index
        pc_cols = [c for c in scores_df.columns if c.startswith('PC')]
        for col in pc_cols:
            alldata_w_pcs.loc[X_pca.index, col] = scores_df[col]
        
        # Reorder columns to put PCs first
        pc_columns_present = [c for c in pc_cols if c in alldata_w_pcs.columns]
        other_columns = [c for c in alldata_w_pcs.columns if c not in pc_columns_present]
        alldata_w_pcs = alldata_w_pcs[pc_columns_present + other_columns]
        
        return {
            "X_pca": X_pca, 
            "ev_df": ev_df, 
            "scores_df": scores_df, 
            "corr_matrix": X_pca.corr(),
            "alldata_w_pcs": alldata_w_pcs
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

        


    # Outputs Multiple Regression
    @render.text
    @reactive.event(input.run_mixedlm)  # only runs when button clicked
    def mixedlm_output():
        import statsmodels.formula.api as smf
        
        # Filter for P glauca
        df_filtered = df[df["species"] == "P glauca"]
        
        # Fit the mixed effects model
        model = smf.mixedlm(
            "change_stem_radius ~ VPD * avg_soil_water_content + avg_solar_irradiance + day_of_year",
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
                change_stem_radius ~ VPD * avg_soil_water_content + avg_solar_irradiance + C(tree) + day_of_year
            """,
            data=df_filtered
        ).fit()
    
    # Return the summary as text
        return str(model.summary())


app = App(app_ui, server)
