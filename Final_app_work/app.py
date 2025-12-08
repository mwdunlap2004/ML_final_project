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



df = pd.read_csv('Final_app_work/categorized_data.csv')
df['Date'] = pd.to_datetime(df['Date']) 

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

df['Date'] = df['Date'].dt.date

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
            * **Logistic Regression**: A predictive model used to classify growth into **"No Growth"**, **"A Little Growth"**, and **"A Lot of Growth"**.
            * **Principal Component Analysis (PCA)**: A dimensionality reduction technique used to explore the structure of the data. It helps us see if variables are correlated and if distinct clusters (like specific Species or Sites) naturally separate in the data.

            ### 3. Variable Definitions
            * **Average Air Temperature**: Measured in Celsius.
            * **Avg Solar Irradiance**: The power per unit area received from the Sun.
            * **Avg Soil Water Content**: Volumetric water content ratio.
            * **Site & Plot**: Categorical identifiers for where the data was collected.
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

    # Tab 3: Logistic Regression Model
    ui.nav_panel(
        "Logistic Regression Model",
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
                ui.h5("What is Logistic Regression?"),
                ui.p("Logistic Regression draws an 'S-shaped' curve to predict the probability of an event happening."),
                ui.h5("Understanding the Outputs"),
                ui.tags.ul(
                    ui.tags.li(ui.strong("Precision"), ": Quality of positive predictions."),
                    ui.tags.li(ui.strong("Recall"), ": Quantity of positives found."),
                    ui.tags.li(ui.strong("Coefficients"), ": Positive values increase likelihood; negative values decrease it.")
                )
            ),
            col_widths=(5, 7)
        ),
        ui.card(
            ui.card_header("Feature Importance: Drivers of 'A Lot of Growth'"),
            ui.output_plot("analysis_plot", height="500px")
        )
    ),

    # Tab 4: PCA Analysis (New)
    ui.nav_panel(
        "PCA Analysis",
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
        ui.card(
            ui.card_header("3. PCA Scatter Plot (PC2 vs PC3)"),
            output_widget("pca_scatter_plot")
        )
    ),
    title="Growth Prediction Dashboard"
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
        # Using fixed features available in your primary dataset
        target = 'growth_category'
        numeric_features = [
            'average_air_temperature', 'average_humidity', 
            'avg_solar_irradiance', 'avg_soil_water_content', 'average_air_pressure'
        ]
        categorical_features = ['species', 'site', 'plot']

        # Verify columns exist before running
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
            ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
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

    # -- C. PCA Logic (New) --
    @reactive.Calc
    def pca_results():
        if df.empty: return None
        
        # Select numeric columns available for PCA
        # We try to use the list from your snippet, but fallback to what is actually in the CSV
        potential_cols = [
            "plot", "average_stem_radius", "change_stem_radius", "average_basal_area", 
            "average_soil_temperature", "average_air_pressure", "average_humidity", 
            "avg_solar_irradiance", "avg_soil_water_content", "freeze_flag",
            # Fallback for old CSV
            "average_air_temperature" 
        ]
        
        # Filter for columns that actually exist in the dataframe and are numeric
        pca_cols = [c for c in potential_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        
        if not pca_cols: return None

        X_pca = df[pca_cols].dropna()
        
        # Pipeline
        pca_pipe = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA())
        ])
        
        pca_pipe.fit(X_pca)
        pca_model = pca_pipe.named_steps["pca"]
        
        # Data for Scree Plot
        explained_var = pca_model.explained_variance_ratio_
        cum_explained_var = np.cumsum(explained_var)
        ev_df = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(len(explained_var))],
            "ExplainedVariance": explained_var,
            "CumulativeVariance": cum_explained_var
        })

        # Transformed Data for Scatter
        X_pca_scores = pca_pipe.transform(X_pca)
        pc_cols = [f"PC{i+1}" for i in range(X_pca_scores.shape[1])]
        scores_df = pd.DataFrame(X_pca_scores, columns=pc_cols, index=X_pca.index)
        
        # Re-attach metadata for hovering (safely)
        if 'species' in df.columns: scores_df['species'] = df.loc[X_pca.index, 'species']
        if 'growth_category' in df.columns: scores_df['Growth'] = df.loc[X_pca.index, 'growth_category']
        
        return {
            "X_pca": X_pca,
            "ev_df": ev_df,
            "scores_df": scores_df,
            "corr_matrix": X_pca.corr()
        }

    # -- Outputs: Data Table --
    @render.data_frame
    def data_table():
        return render.DataTable(filtered_df())

    # -- Outputs: Logistic Regression --
    @render.text
    def accuracy_score():
        res = model_results()
        return f"{res['score']:.4f}" if res else "No data"

    @render.text
    def class_report():
        res = model_results()
        return res['report_txt'] if res else "No data"

    @render.plot
    def analysis_plot():
        res = model_results()
        if not res: return
        # ... (Existing Matplotlib code for Coefficients/Performance) ...
        # Simplified for brevity in this snippet, assumes previous logic matches
        model = res["model"]
        report_dict = res["report_dict"]
        num_feats = res["numeric_features"]
        cat_feats = res["categorical_features"]
        
        # 1. Coefficients
        cat_names = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_feats)
        feat_names = num_feats + list(cat_names)
        target_cls = 'A Lot of Growth'
        
        if target_cls in model.classes_:
            idx = list(model.classes_).index(target_cls)
            coefs = model.named_steps['classifier'].coef_[idx]
            df_coef = pd.DataFrame({'Feature': feat_names, 'Coefficient': coefs})
            df_coef = df_coef.reindex(df_coef.Coefficient.abs().sort_values(ascending=False).index).head(5)
        else:
            df_coef = pd.DataFrame({'Feature': ['N/A'], 'Coefficient': [0]})

        # 2. Performance
        classes = [c for c in ['A Little Growth', 'A Lot of Growth', 'No Growth'] if c in report_dict]
        df_perf = pd.DataFrame({
            'Class': classes,
            'Precision': [report_dict[c]['precision'] for c in classes],
            'Recall': [report_dict[c]['recall'] for c in classes]
        }).melt(id_vars='Class', var_name='Metric', value_name='Score')

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.barplot(x='Coefficient', y='Feature', data=df_coef, ax=axes[0], palette='viridis')
        axes[0].set_title(f"Top 5 Drivers of '{target_cls}'")
        
        if not df_perf.empty:
            sns.barplot(x='Class', y='Score', hue='Metric', data=df_perf, ax=axes[1], palette='RdBu')
            axes[1].set_title("Model Performance")
            axes[1].set_ylim(0, 1.0)
            axes[1].axhline(0.33, color='red', linestyle='--')
        
        plt.tight_layout()
        return fig

    # -- Outputs: PCA (Plotly) --
    @render_widget
    def corr_matrix_plot():
        res = pca_results()
        if not res: return go.Figure()
        return px.imshow(
            res["corr_matrix"], text_auto=True, color_continuous_scale="RdBu_r", aspect="auto",
            title="Correlation Matrix"
        )

    @render_widget
    def scree_plot():
        res = pca_results()
        if not res: return go.Figure()
        return px.line(
            res["ev_df"], x="PC", y="ExplainedVariance", markers=True,
            title="Scree Plot: Proportion of Variance Explained"
        )

    @render_widget
    def pca_scatter_plot():
        res = pca_results()
        if not res: return go.Figure()
        sdf = res["scores_df"]
        
        # Check if we have enough PCs to plot PC2 vs PC3
        x_col = "PC2" if "PC2" in sdf.columns else "PC1"
        y_col = "PC3" if "PC3" in sdf.columns else "PC2"
        
        # Color by growth if available, else species
        color_col = "Growth" if "Growth" in sdf.columns else (
            "species" if "species" in sdf.columns else None
        )

        return px.scatter(
            sdf, x=x_col, y=y_col, color=color_col,
            hover_data=sdf.columns,
            title=f"PCA Scatter: {x_col} vs {y_col}",
            opacity=0.7, height=600
        )

app = App(app_ui, server)