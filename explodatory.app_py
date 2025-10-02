import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA App", layout="wide")

# --------------------------------
# 1. File Upload
# --------------------------------
st.title("📊 Exploratory Data Analysis (EDA) App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success("✅ File successfully uploaded!")
    
    # Show Dataset
    st.subheader("📄 Dataset Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # --------------------------------
    # 2. Dataset Info
    # --------------------------------
    with st.expander("🔎 Dataset Info"):
        st.write("**Column Names:**", df.columns.tolist())
        st.write("**Data Types:**")
        st.write(df.dtypes)
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
        st.write("**Duplicates:**", df.duplicated().sum())

    # --------------------------------
    # 3. Univariate Analysis
    # --------------------------------
    with st.expander("📈 Univariate Analysis"):
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include="object").columns

        st.markdown("### Numerical Columns")
        st.write(df[num_cols].describe())

        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        st.markdown("### Categorical Columns")
        for col in cat_cols:
            fig, ax = plt.subplots()
            sns.countplot(x=col, data=df, ax=ax)
            ax.set_title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # --------------------------------
    # 4. Bivariate Analysis
    # --------------------------------
    with st.expander("🔗 Bivariate Analysis"):
        if len(num_cols) > 1:
            st.markdown("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        st.markdown("### Numerical vs Categorical")
        for cat in cat_cols:
            for num in num_cols:
                fig, ax = plt.subplots()
                sns.boxplot(x=cat, y=num, data=df, ax=ax)
                ax.set_title(f"{num} by {cat}")
                plt.xticks(rotation=45)
                st.pyplot(fig)

    # --------------------------------
    # 5. Outlier Detection
    # --------------------------------
    with st.expander("⚠️ Outlier Detection"):
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Outlier Check for {col}")
            st.pyplot(fig)

    # --------------------------------
    # 6. Feature Engineering Example
    # --------------------------------
    with st.expander("🛠 Feature Engineering Examples"):
        if "Sales" in df.columns:
            df["Log_Sales"] = np.log1p(df["Sales"])
            st.write("Added `Log_Sales` column (log-transformed Sales).")
            st.write(df[["Sales", "Log_Sales"]].head())

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            st.write("Extracted `Year` and `Month` from Date column.")
            st.write(df[["Date", "Year", "Month"]].head())

    # --------------------------------
    # 7. Download Cleaned Data
    # --------------------------------
    with st.expander("💾 Download Cleaned Data"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "cleaned_dataset.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file to start EDA.")
