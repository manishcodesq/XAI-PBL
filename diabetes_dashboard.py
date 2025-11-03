# diabetes_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import plotly.express as px
import traceback
import sys

# MUST be the first Streamlit call
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Main App
# ---------------------------------------------------
def main():
    st.title("🩺 Diabetes Prediction Using XAI - Dashboard")
    st.write("Welcome! This app demonstrates **XAI methods** like LIME & SHAP "
             "to interpret ML models on diabetes prediction.")

    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Dataset", "Visualizations", "Model Explanation"])

    # Load dataset (Pima Indians Diabetes Dataset from sklearn)
    from sklearn.datasets import load_diabetes
    data = load_diabetes(as_frame=True)
    df = data.frame

    if page == "Home":
        st.subheader("About this Dashboard")
        st.markdown("""
        - Built with **Streamlit**
        - Uses **LIME** and **SHAP** for explainable AI  
        - Example dataset: Pima Indians Diabetes  
        """)

        # 👥 Team Members
        st.subheader("👥 Project Team")
        team_data = {
            "PRN": ["22310149", "22311496", "22311608", "22311611"],
            "Roll No": ["311005", "311049", "311058", "311059"],
            "Name": ["Manish Kulkarni", "Atharva Joshi", "Manas Kulkarni", "Arnav Kulkarni"]
        }
        df_team = pd.DataFrame(team_data)

        st.table(df_team)


    elif page == "Dataset":
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("**Shape of dataset:**", df.shape)
        st.write("**Columns:**", list(df.columns))

    elif page == "Visualizations":
        st.subheader("Visualizations")

        col = st.selectbox("Select feature to visualize", df.columns)
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Model Explanation":
        st.subheader("Model Explanation (Demo)")

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor

        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        st.write("### SHAP Explanation")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:50])

        st.write("Feature importance (SHAP summary plot):")
        shap_fig = shap.summary_plot(shap_values, X_test[:50], plot_type="bar", show=False)
        st.pyplot(shap_fig, clear_figure=True)

        st.write("### LIME Explanation")
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            mode="regression"
        )

        i = 1
        exp = explainer_lime.explain_instance(X_test.iloc[i], model.predict, num_features=5)
        st.write(exp.as_list())
        st.pyplot(exp.as_pyplot_figure())

# ---------------------------------------------------
# Entry point
# ---------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        try:
            st.error("⚠️ Fatal error occurred — check terminal for details.")
        except Exception:
            pass
