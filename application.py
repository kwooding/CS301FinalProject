import base64
import io

import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
from dash.dash_table.Format import Group
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dash Model Training App"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Upload and Select Target")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a CSV File')
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                },
                multiple=False,
                accept=".csv"
            ),
            html.Div(id='upload-status', style={'margin': '10px', 'color': 'green'}),
            dcc.Store(id='stored-data'),
            dcc.Store(id='stored-model')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select Target Variable"),
            dcc.Dropdown(id='target-dropdown')
        ])
    ]),

    html.Hr(),

    # Bar Charts
    dbc.Row([
        dbc.Col([
            html.H3("Average Target by Categorical Variable"),
            dcc.RadioItems(id='categorical-radio', inputStyle={"margin-right": "5px"}),
            dcc.Graph(id='avg-target-chart')
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Correlation with Numerical Variables"),
            dcc.Graph(id='correlation-chart')
        ])
    ]),

    html.Hr(),

    # Train Section
    html.H2("Train"),
    html.Label("Select Features:"),
    dcc.Checklist(id='feature-checklist', inline=True),
    html.Br(),
    html.Button("Train Model", id='train-button', n_clicks=0),
    html.Div(id='train-output', style={'margin': '10px', 'color': 'green'}),

    html.Hr(),

    # Predict Section
    html.H2("Predict"),
    html.Div("Enter values in order of how they were selected in the training model. If categorical, enter the category exactly as it appears in the dataset."),
    dcc.Input(id='predict-input', type='text', placeholder='comma-separated values'),
    html.Button("Predict", id='predict-button', n_clicks=0),
    html.Div(id='prediction-output', style={'margin': '10px', 'color': 'green'})
], fluid=True)


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    columns_to_drop = ['Resources_used', 'Duration_to_save(in_Years) ', 'Goal_for_investment']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    return df

@app.callback(
    Output('upload-status', 'children'),
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        return "Dataset uploaded and cleaned.", df.to_dict('records')
    return "", None

@app.callback(
    Output('target-dropdown', 'options'),
    Output('target-dropdown', 'value'),
    Input('stored-data', 'data')
)
def update_target_options(data):
    if data is None:
        return [], None
    df = pd.DataFrame(data)
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_columns) > 0:
        return [{'label': col, 'value': col} for col in numerical_columns], numerical_columns[0]
    else:
        return [], None

@app.callback(
    Output('categorical-radio', 'options'),
    Output('categorical-radio', 'value'),
    Input('stored-data', 'data'),
    Input('target-dropdown', 'value')
)
def update_categorical_radio(data, target_column):
    if data is None or target_column is None:
        return [], None
    df = pd.DataFrame(data)
    categorical_columns = df.select_dtypes(include=[object]).columns.tolist()
    if len(categorical_columns) > 0:
        return [{'label': col, 'value': col} for col in categorical_columns], categorical_columns[0]
    return [], None

@app.callback(
    Output('avg-target-chart', 'figure'),
    Input('stored-data', 'data'),
    Input('target-dropdown', 'value'),
    Input('categorical-radio', 'value')
)
def update_avg_target_chart(data, target_column, selected_cat_col):
    if data is None or target_column is None or selected_cat_col is None:
        return {}
    df = pd.DataFrame(data)
    avg_data = df.groupby(selected_cat_col)[target_column].mean().reset_index()
    # Limit categories to top 10 by target
    max_categories = 10
    if len(avg_data) > max_categories:
        avg_data = avg_data.nlargest(max_categories, target_column)

    fig = px.bar(avg_data, x=selected_cat_col, y=target_column, title=f"Average {target_column} by {selected_cat_col}")
    fig.update_xaxes(tickangle=45)
    return fig

@app.callback(
    Output('correlation-chart', 'figure'),
    Input('stored-data', 'data'),
    Input('target-dropdown', 'value')
)
def update_correlation_chart(data, target_column):
    if data is None or target_column is None:
        return px.bar(title="No data or target selected.")
    try:
        df = pd.DataFrame(data)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column not in numeric_columns or len(numeric_columns) == 0:
            return px.bar(title="No numeric columns available for correlation.")

        # Compute correlation and exclude the target variable
        corr_series = df[numeric_columns].corr()[target_column].abs().sort_values(ascending=False)
        corr_series = corr_series[corr_series.index != target_column]

        if corr_series.empty:
            return px.bar(title="No correlations to display.")

        corr_df = pd.DataFrame({'Feature': corr_series.index, 'Correlation': corr_series.values})
        fig = px.bar(corr_df, x='Feature', y='Correlation',
                     title=f"Correlation Strength of Numerical Variables with {target_column}")
        fig.update_xaxes(tickangle=45)
        return fig
    except Exception as e:
        return px.bar(title=f"Error: {str(e)}")
def update_correlation_chart(data, target_column):
    if data is None or target_column is None:
        return {}
    try:
        df = pd.DataFrame(data)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column not in numeric_columns or len(numeric_columns) == 0:
            return {}
        # Compute correlation and exclude the target variable
        corr_series = df[numeric_columns].corr()[target_column].abs().sort_values(ascending=False)
        corr_series = corr_series[corr_series.index != target_column]

        corr_df = pd.DataFrame({'Feature': corr_series.index, 'Correlation': corr_series.values})
        fig = px.bar(corr_df, x='Feature', y='Correlation',
                     title=f"Correlation Strength of Numerical Variables with {target_column}")
        fig.update_xaxes(tickangle=45)
        return fig
    except Exception as e:
        return px.bar(title=f"Error: {str(e)}")
def update_correlation_chart(data, target_column):
    if data is None or target_column is None:
        return {}
    df = pd.DataFrame(data)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column not in numeric_columns or len(numeric_columns) == 0:
        return {}
    # Compute correlation and exclude the target variable
    corr_series = df[numeric_columns].corr()[target_column].abs().sort_values(ascending=False)
    corr_series = corr_series[corr_series.index != target_column]

    corr_df = pd.DataFrame({'Feature': corr_series.index, 'Correlation': corr_series.values})
    fig = px.bar(corr_df, x='Feature', y='Correlation',
                 title=f"Correlation Strength of Numerical Variables with {target_column}")
    fig.update_xaxes(tickangle=45)
    return fig

@app.callback(
    Output('feature-checklist', 'options'),
    Output('feature-checklist', 'value'),
    Input('stored-data', 'data'),
    Input('target-dropdown', 'value')
)
def update_feature_checklist(data, target_column):
    if data is None or target_column is None:
        return [], []
    df = pd.DataFrame(data)
    features = [col for col in df.columns if col != target_column]
    options = [{'label': f" {col}", 'value': col} for col in features]
    return options, []


trained_models = {}

@app.callback(
    Output('train-output', 'children'),  # Remove the `stored-model.data` output
    Input('train-button', 'n_clicks'),
    State('feature-checklist', 'value'),
    State('stored-data', 'data'),
    State('target-dropdown', 'value')
)
def train_model(n_clicks, selected_features, data, target_column):
    if n_clicks > 0:
        if data is None or target_column is None or len(selected_features) == 0:
            return "Please select a dataset, target, and at least one feature before training."
        df = pd.DataFrame(data)
        X = df[selected_features]
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define preprocessing pipelines
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [col for col in selected_features if col not in numeric_features]

        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Train the model
        rf_model = RandomForestRegressor(random_state=23)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', rf_model)
        ])

        pipeline.fit(X_train, y_train)

        # Save the trained model in the global dictionary
        trained_models['model'] = pipeline

        # Evaluate the model
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        return f"Random Forest R²: {r2:.2f}"

    return ""

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('predict-input', 'value'),
    State('feature-checklist', 'value'),
    State('stored-data', 'data'),
    State('target-dropdown', 'value')
)
def predict_values(n_clicks, feature_input, selected_features, data, target_column):
    if n_clicks > 0:
        # Ensure a model has been trained
        if 'model' not in trained_models:
            return "Model is not trained yet. Please train the model first."

        if not feature_input:
            return "Please enter feature values to make a prediction."

        # Parse input
        pipeline = trained_models['model']
        df = pd.DataFrame(data)
        values = feature_input.split(",")
        if len(values) != len(selected_features):
            return "Please enter the correct number of feature values."

        feature_values = []
        # Determine if feature is numeric or categorical
        for col, val in zip(selected_features, values):
            if col in df.select_dtypes(include=[np.number]).columns.tolist():
                try:
                    feature_values.append(float(val.strip()))
                except ValueError:
                    return f"Value '{val}' for feature '{col}' is not numeric."
            else:
                # Check case-insensitive matching for categorical variables
                unique_values = df[col].str.lower().unique()
                val_lower = val.strip().lower()
                if val_lower in unique_values:
                    # Add the value with original case from the dataset
                    original_value = df[col][df[col].str.lower() == val_lower].iloc[0]
                    feature_values.append(original_value)
                else:
                    return f"Value '{val}' for categorical feature '{col}' is not found in the dataset. Please enter again."

        input_df = pd.DataFrame([feature_values], columns=selected_features)
        prediction = pipeline.predict(input_df)[0]

        # Only round if the target_column is "Working_Professional"
        if target_column == 'Working_Professional':
            prediction = 1 if prediction > 0.5 else 0

        return f"Predicted {target_column}: {prediction}"

    return ""


if __name__ == "__main__":
    app.run_server(debug=False)
