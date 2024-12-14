import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Global variables
df = None
model = None
selected_features = []

# Upload Component
def upload_component():
    global df
    st.title("Upload and Select Target")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        columns_to_drop = ['Resources_used', 'Duration_to_save(in_Years) ', 'Goal_for_investment']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        st.success("Dataset uploaded and cleaned.")

# Select Target Component
def select_target_component():
    global df
    if df is not None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        target_column = st.selectbox("Select Target Variable", numerical_columns)
        return target_column
    else:
        st.warning("Please upload a dataset first.")
        return None

# Bar Charts Component
def barcharts_component(target_column):
    global df
    if df is not None and target_column:

        # First Bar Chart: Average Target by Categorical Variable
        st.subheader("Average Target by Categorical Variable")
        categorical_columns = df.select_dtypes(include=[object]).columns.tolist()
        selected_cat_col = st.radio("Choose a categorical column", categorical_columns, horizontal=True)

        avg_data = df.groupby(selected_cat_col)[target_column].mean().reset_index()
        
        # Limiting the number of categories shown (Top N categories)
        max_categories = 10
        if len(avg_data) > max_categories:
            avg_data = avg_data.nlargest(max_categories, columns=target_column)

        fig, ax = plt.subplots(figsize=(8, 6))  # Consistent size
        sns.barplot(x=selected_cat_col, y=target_column, data=avg_data, ax=ax)
        ax.set_title(f"Average {target_column} by {selected_cat_col}")

        # Renaming X-axis
        ax.set_xticklabels(avg_data[selected_cat_col], rotation=45, ha='right')
        st.pyplot(fig)

        # Second Bar Chart: Correlation with Numerical Variables
        st.subheader("Correlation with Numerical Variables")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        correlation_data = df[numeric_columns].corr()[target_column].abs().sort_values(ascending=False)
        correlation_data = correlation_data[correlation_data.index != target_column]

        fig, ax = plt.subplots(figsize=(8, 6))  # Consistent size
        sns.barplot(x=correlation_data.index, y=correlation_data.values, ax=ax, color="blue")
        ax.set_title(f"Correlation Strength of Numerical Variables with {target_column}")
        ax.set_ylabel("Correlation Strength (Absolute Value)")
        ax.set_xlabel("Numerical Variables")
        ax.set_xticklabels(correlation_data.index)
        st.pyplot(fig)

# Train Component
def train_component(target_column):
    global df, selected_features, model
    if df is not None and target_column:
        st.header("Train")

        # Persistent feature selection
        st.subheader("Select Features")
        features = [col for col in df.columns if col != target_column]
        selected_features = [
            feature for feature in features if st.checkbox(feature, key=f"checkbox_{feature}")
        ]

        if selected_features:
            X = df[selected_features]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = [col for col in selected_features if col not in numeric_features]

            numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                                      ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)])

            # Train button triggers training
            train_button = st.button("Train Model")
            if train_button:
                rf_model = RandomForestRegressor(random_state=23)
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', rf_model)
                ])

                pipeline.fit(X_train, y_train)

                st.session_state['model'] = pipeline

                # Evaluate performance
                y_pred = st.session_state['model'].predict(X_test)
                r2 = r2_score(y_test, y_pred)

                st.success(f"Random Forest R²: {r2:.2f}")

        else:
            st.warning("Please select at least one feature to proceed with training.")

def predict_component(target_column):
    global selected_features
    if 'model' in st.session_state and st.session_state['model'] is not None:
        st.header("Predict")
        
        st.text("Enter values in order of how they are selected in training model. "
                "See visualization graphs for categorical names and enter exactly as shown.")

        feature_input = st.text_input("Enter feature values (comma-separated)")
        predict_button = st.button("Predict")

        if predict_button:
            if feature_input:
                try:
                    values = feature_input.split(",")
                    feature_values = []

                    for col, value in zip(selected_features, values):
                        if col in df.select_dtypes(include=[np.number]).columns:
                            # Handle numerical features
                            feature_values.append(float(value.strip()))
                        else:
                            # Handle categorical features
                            feature_values.append(value.strip())

                    if len(feature_values) == len(selected_features):
                        input_df = pd.DataFrame([feature_values], columns=selected_features)
                        prediction = st.session_state['model'].predict(input_df)[0]
                        st.success(f"Predicted {target_column}: {prediction:.2f}")
                    else:
                        st.error("Please enter the correct number of feature values.")

                except ValueError:
                    st.error("Please enter valid values.")
            else:
                st.warning("Please enter feature values to make a prediction.")
    else:
        st.warning("Model is not trained yet. Please train the model first.")


# Main App
def main():
    upload_component()
    target_column = select_target_component()

    if target_column:
        barcharts_component(target_column)
        train_component(target_column)
        predict_component(target_column)

if __name__ == "__main__":
    main()