import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Model related functions goes here

def preprocess_n_dr(df):
    # drop the cx number
    df_d = df.drop(['MSISDN_ENCR_INT'], axis=1)

    #fill null values
    df_d = df_d.fillna(0)

    # initialize the standardscaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_d)

    # # Determine max possible PCA components
    # max_components = min(df_scaled.shape[0], df_scaled.shape[1])  # min(samples, features)

    # # Apply PCA with max allowed components
    # n_pca_components = min(18, max_components)

    # apply PCA
    pca_opt = PCA(n_components=18)
    df_pca_reduced = pca_opt.fit_transform(df_scaled)

    # Convert PCA result back to a DataFrame
    pca_columns = [f"PC{i+1}" for i in range(df_pca_reduced.shape[1])]
    df_pca = pd.DataFrame(df_pca_reduced, columns=pca_columns)

    return df_pca

# This is a dummy function that expects the data frame,run the model and output probability
def predict(df, df_original, th=0.617):
    # Load the trained model
    churn_model = joblib.load('model/churn_prediction_model.pkl')
        
    # Make predictions
    predictions = churn_model.predict_proba(df)[:, 1]
        
    # Apply threshold to get churn tag
    predicted_tag = (predictions >= th).astype(int)
        
    # Add predictions and churn tags to the original DataFrame
    df_original = df_original.copy()
    df_original['PREDICTION'] = predictions
    df_original['CHURN_TAG'] = predicted_tag
        
    return df_original

    # data = {
    #     "customer_no": range(1, 11),
    #     "category": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
    #     "churn": [True, False, True, False, True, False, True, False, True, False]}
    # return pd.DataFrame(df_original)

# Utility functions
def show_summary_stats(df):
    # Number of records in the file
    total_customers = len(df)

    # Calculate duplicate records
    duplicate_records = df.duplicated().sum()

    # Average Revenue Per User (ARPU)
    l3M_ARPU_per_customer = df['L3M_ARPU'].mean()

    # Average usage per user
    usage_columns = ['L30_DATA_USAGE', 'L30_DATA_PAYGO_USAGE', 'L30_VOICE_USAGE']
    df['TOTAL_USAGE'] = df[usage_columns].sum(axis=1)
    avg_usage_per_customer = df['TOTAL_USAGE'].mean()

    # Display the KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{total_customers}")
    col2.metric("Duplicate Records", f"{duplicate_records}")
    col3.metric("ARPU per Customer", f"{l3M_ARPU_per_customer:,.1f}")
    col4.metric("Avg Usage per Customer (voice and data)", f"{avg_usage_per_customer:,.1f}")

# Updated function to display summary statistics with boxed values
# def show_summary_stats(df: pd.DataFrame):
#     # Calculate key metrics
#     total_customers = len(df)
#     duplicate_records = df.duplicated().sum()
#     l3M_ARPU_per_customer = df['L3M_ARPU'].mean()
#     usage_columns = ['L30_DATA_USAGE', 'L30_DATA_PAYGO_USAGE', 'L30_VOICE_USAGE']
#     df['TOTAL_USAGE'] = df[usage_columns].sum(axis=1)
#     avg_usage_per_customer = df['TOTAL_USAGE'].mean()


#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         st.markdown(f'''<div style="border:1px solid #ddd; padding:10px; border-radius:8px;">
#                           <h4 style="margin:0;">Total Records</h4>
#                           <p style="font-size:1.5em; font-weight:bold;">{total_customers:,}</p>
#                       </div>''', unsafe_allow_html=True)

#     with col2:
#         st.markdown(f'''<div style="border:1px solid #ddd; padding:10px; border-radius:8px;">
#                           <h4 style="margin:0;">Duplicate Records</h4>
#                           <p style="font-size:1.5em; font-weight:bold;">{duplicate_records:,}</p>
#                       </div>''', unsafe_allow_html=True)

#     with col3:
#         st.markdown(f'''<div style="border:1px solid #ddd; padding:10px; border-radius:8px;">
#                           <h4 style="margin:0;">ARPU per Customer</h4>
#                           <p style="font-size:1.5em; font-weight:bold;">Rs. {l3M_ARPU_per_customer:,.1f}</p>
#                       </div>''', unsafe_allow_html=True)

#     with col4:
#         st.markdown(f'''<div style="border:1px solid #ddd; padding:10px; border-radius:8px;">
#                           <h4 style="margin:0;">Avg Usage per Customer (Voice & Data)</h4>
#                           <p style="font-size:1.5em; font-weight:bold;">{avg_usage_per_customer:,.1f}</p>
#                       </div>''', unsafe_allow_html=True)

#     st.write("---")


def draw_pie_chart(df, column='CHURN_TAG', filter_column='None'):
    # Calculate the counts of churned (1) and non-churned (0)
    custom_labels = {
            'CHURN_TAG': {0: 'Non-Churned', 1: 'Churned'},
            'SMARTPHONE_USER': {0: 'Non-Smartphone User', 1: 'Smartphone User'}
        }

    if filter_column == 'None':
        counts = df[column].value_counts()
        if column in custom_labels:
            labels = [f"{custom_labels[column].get(label, label)} ({count})" for label, count in counts.items()]
        else:
            labels = [f"{label} ({count})" for label, count in counts.items()]
        sizes = counts.values
        
        # Create the pie chart
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, colors=plt.cm.Paired.colors)
        ax1.axis('equal')  # Equal aspect ratio ensures the pie is a circle.
        
        # Display the chart in Streamlit
        st.pyplot(fig1)

    else:
        df_display = df[df[filter_column] == 1]
        counts = df_display[column].value_counts()
        if column in custom_labels:
            labels = [f"{custom_labels[column].get(label, label)} ({count})" for label, count in counts.items()]
        else:
            labels = [f"{label} ({count})" for label, count in counts.items()]
        sizes = counts.values
        
        # Create the pie chart
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, colors=plt.cm.Paired.colors)
        ax1.axis('equal')  # Equal aspect ratio ensures the pie is a circle.
        
        # Display the chart in Streamlit
        st.pyplot(fig1)

def draw_bar_chart(df, column='CHURN_TAG', filter_column=None):
    
    # Define bins based on the column passed
    if column == 'AON' and filter_column != 'None':
        df_display = df[df[filter_column] == 1]
        df_display['AON_MONTHS'] = df_display[column] / 30
        bins = [0, 3, 6, 12, np.inf]  # Example bin ranges for AON (in months)
        labels = ['0-3 Months', '3-6 Months', '6-12 Months', '12 Months+']  # Bin labels
        df_display['BINS'] = pd.cut(df_display['AON_MONTHS'], bins=bins, labels=labels)
        bin_counts = df_display['BINS'].value_counts().sort_index()

        # Plot the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bin_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)

        # Annotate bin counts on top of each bar
        for i, count in enumerate(bin_counts):
            ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12)  # Adjust 0.5 for spacing

        # Customize the chart
        # ax.set_title(f"Distribution of {column} Binned Data", fontsize=16)
        ax.set_xlabel("Network Stay", fontsize=14)
        ax.set_ylabel("Number of Customers", fontsize=14)
        ax.set_xticklabels(bin_counts.index, rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Display the plot in Streamlit
        st.pyplot(fig)
    else:
        # Define custom bins and labels for other columns if necessary
        # Example for other column values, you can customize this
        bins = [0, 100, 200, 300, 400, np.inf]
        labels = ['0-100', '100-200', '200-300', '300-400', '400+']

    

    

def analyze_results(df):
    churned_customers = df[df['CHURN_TAG'] == 1].shape[0]
    non_churn = len(df) - churned_customers

    churn_count, non_churn_count, void1, void2 = st.columns(4)
    churn_count.metric("Churn Predicted", f"{churned_customers}")
    non_churn_count.metric("Non Churn Predicted", f"{non_churn}")

    # Create horizontal bar chart
    # fig = go.Figure()

    # fig.add_trace(go.Bar(
    #     y=churn_counts.index, 
    #     x=churn_counts[True], 
    #     name="Churn", 
    #     orientation="h",
    #     marker_color="red"
    # ))

    # fig.add_trace(go.Bar(
    #     y=churn_counts.index, 
    #     x=churn_counts[False], 
    #     name="Non-Churn", 
    #     orientation="h",
    #     marker_color="green"
    # ))

    # fig.update_layout(
    #     title="Churn vs Non-Churn by Category",
    #     xaxis_title="Count",
    #     yaxis_title="Category",
    #     barmode="group"  # Bars will be side-by-side
    # )

    # st.plotly_chart(fig)


@st.cache_data
def convert_df(df):
    #Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

def to_results_csv(df):
    # TODO : do whatever post-processing needed here

    return convert_df(df)

# APP Code starts here
TITLE = "Customer Churn Prediction"
OUTPUT_FILE_NAME = "predict_results.csv"

st.set_page_config(layout="wide")

# configuring title bar
st.title(TITLE)
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# side bar
uploaded_file = st.sidebar.file_uploader("Choose the input file (supported formats : csv)")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_preprocessed = preprocess_n_dr(df)
    results_df = predict(df_preprocessed, df, th=0.617)

    # 1. General statistics
    # Header for Summary Stats
    st.write("### Summary Statistics")
    show_summary_stats(df)

    # 2. Pie chart for churn/not churn
    st.write("### Prediction Results")

    # 2. churn anaylysis
    analyze_results(results_df)

    col1_1, col2_2 = st.columns(2)
    with col1_1:
        st.write("##### Churn - Network Stay Distribution")
        # draw_pie_chart(results_df, column='CHURN_TAG',filter_column='None')
        draw_bar_chart(results_df, column='AON', filter_column='CHURN_TAG')
    with col2_2:
        st.write("##### Churned - Smartphone vs Non Smartphone")
        draw_pie_chart(results_df, column='SMARTPHONE_USER', filter_column='CHURN_TAG')

    # 3. download results
    result_csv = to_results_csv(results_df)
    st.download_button(label="Download Results",data=result_csv, file_name= OUTPUT_FILE_NAME,mime="text/csv")
