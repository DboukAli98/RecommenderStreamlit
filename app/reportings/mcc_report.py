import os
from pathlib import Path
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import ollama
import boto3
from dotenv import load_dotenv
from io import StringIO




# Streamlit UI
st.set_page_config(page_title="MCC Score Dashboard", layout="wide")


# Loading environment variables
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME= os.getenv("AWS_BUCKET_NAME")

#s3 client initialization
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)



#setting dynamic directories

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
base_dir = parent_dir.parent

data_dir = base_dir / "data"


#loading data 

@st.cache_data
def load_data():
    
    file1 = "mcc_scores.csv"
    file2 = "rewards_transactions_cleaned.csv"
    file3 = "new_segments.csv"
    
    response1 = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=file1)
    df = pd.read_csv(StringIO(response1['Body'].read().decode('utf-8')))
    df = df.drop(columns=["Unnamed: 0"])
    
    response2 = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=file2)
    df2 = pd.read_csv(StringIO(response2['Body'].read().decode('utf-8')))
    
    response3 = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=file3)
    
    new_segments = pd.read_csv(StringIO(response3['Body'].read().decode('utf-8')))
    return df , df2 , new_segments

df , df2 , new_segments = load_data()



# response = ollama.chat(
#     model='llama3.2:latest',
#     messages=[
#         {"role": "system", "content": "You are a data analyst that explains and interprets charts , dataframes to user.The data are for the TAM loyalty application"},
#         {"role": "user", "content": f"Analyze the following DataFrame:\n{df.to_string()}"}
#     ]
# )

# print(response['message']['content'])





st.title("📊 MCC Score Analysis Dashboard")

# Sidebar Filters for MCC
st.sidebar.header("Filters")

# Getting Top 5 & Top 10 MCCs based on TotalPointsRewarded
top_5_mccs = df.nlargest(5, "TotalPointsRewarded")["MCC"].tolist()
top_10_mccs = df.nlargest(10, "TotalPointsRewarded")["MCC"].tolist()


# Defining selection options
all_mccs = df["MCC"].unique().tolist()
all_option = ["Select All", "Top 5", "Top 10"] + all_mccs

# Multiselect with additional options
selected_mccs = st.sidebar.multiselect("Select MCC(s):", all_option, default=["Top 10"])

# Apply selection logic
if "Select All" in selected_mccs:
    selected_mccs = all_mccs
elif "Top 5" in selected_mccs:
    selected_mccs = top_5_mccs
elif "Top 10" in selected_mccs:
    selected_mccs = top_10_mccs

# Filtering the Data
filtered_df = df[df["MCC"].isin(selected_mccs)]


# Displaying the Data Table
st.subheader("MCC Score Data")
st.dataframe(filtered_df)


# Pie Chart for MCC Distribution
st.subheader("MCC Distribution (Based on Total Points Rewarded)")
fig_pie = px.pie(filtered_df, names="MCC", values="TotalPointsRewarded",
                 title="MCC Distribution by Total Points Rewarded",
                 hole=0.4)
st.plotly_chart(fig_pie)


# Scatter Plot plotting
st.subheader("MCC Score vs Transaction Frequency")
fig = px.scatter(filtered_df, x="TransactionFrequency", y="MCC_Score_Scaled",
                 size="TotalPointsRewarded", color="MCC",
                 hover_data=["MCC", "TotalPointsRewarded"],
                 title="MCC Score vs Transaction Frequency")
st.plotly_chart(fig)

# Bar Chart plotting
st.subheader("Top MCCs by Score")
fig_bar = px.bar(filtered_df, x="MCC", y="MCC_Score_Scaled", color="MCC_Score_Scaled",
                 title="Top MCCs by Score", labels={"MCC": "MCC Category", "MCC_Score_Scaled": "MCC Score"})
st.plotly_chart(fig_bar)

st.sidebar.markdown("#### Note the score is calculated with the follwing formula : MCC Score = Total Points Reward x Transaction Frequency / max(total Points Rewarded x Transaction Frequency)")


mcc_tier_grouped = df2.groupby(["Cluster_Name", "MCC"]).agg(
    TotalPointsRewarded=("PointsRewarded", "sum"),
    TransactionFrequency=("TrxId", "count")
).reset_index()


mcc_tier_grouped = mcc_tier_grouped.dropna(subset=["Cluster_Name", "MCC"])


mcc_tier_grouped["Cluster_Name"] = mcc_tier_grouped["Cluster_Name"].astype(str)


mcc_tier_grouped["MCC_Score"] = (
    mcc_tier_grouped["TotalPointsRewarded"] * mcc_tier_grouped["TransactionFrequency"]
)
mcc_tier_grouped["MCC_Score"] /= mcc_tier_grouped.groupby("Cluster_Name")["MCC_Score"].transform("max")
mcc_tier_grouped = mcc_tier_grouped.dropna(subset=["MCC_Score"])

top_mcc_per_tier = mcc_tier_grouped.loc[mcc_tier_grouped.groupby("Cluster_Name")["MCC_Score"].idxmax()]


st.subheader("📈 Users Segments")
st.dataframe(new_segments)

st.caption("**Note :** The segments are based on five key features that describe user spending behavior as follow :\n\n **Total Amount Spent** , **Total Points Rewarded** , **Transaction Frequency** , **Average Spending Per Transaction** , **Average Points Per Transaction**  ")

st.subheader("🏆 Top MCC by Segments")
st.dataframe(top_mcc_per_tier)





st.subheader("🍰 Highest MCC Score Distribution by Segments")
fig_pie = px.pie(
    top_mcc_per_tier, names="MCC", values="MCC_Score", title="Top MCCs per Segments"
)
st.plotly_chart(fig_pie)



st.sidebar.header("Segments Filters")
tier_options = ["All"] + sorted(mcc_tier_grouped["Cluster_Name"].unique().tolist()) 
selected_tier = st.sidebar.selectbox("Select Segment:", tier_options)


if selected_tier == "All":
    filtered_mcc_tier = mcc_tier_grouped
else:
    filtered_mcc_tier = mcc_tier_grouped[mcc_tier_grouped["Cluster_Name"] == selected_tier]


st.subheader(f"🏆 Top MCCs for {selected_tier if selected_tier != 'All' else 'All Segments'}")
st.dataframe(filtered_mcc_tier.sort_values(by="MCC_Score", ascending=False).head(10))


st.subheader(f"📊 MCC Score Distribution for {selected_tier if selected_tier != 'All' else 'All Segments'}")
fig_bar = px.bar(
    filtered_mcc_tier.sort_values(by="MCC_Score", ascending=False).head(10), 
    x="MCC", y="MCC_Score", color="MCC_Score",
    title=f"Top MCCs by Score in {selected_tier if selected_tier != 'All' else 'All Segments'}",
    labels={"MCC": "MCC Category", "MCC_Score": "MCC Score"}
)
st.plotly_chart(fig_bar)

from sklearn.preprocessing import MinMaxScaler

def get_top_stores_with_composite_score(df, top_n=3, weights=(1/3, 1/3, 1/3)):
    
    """
    Get the top stores in each MCC using a composite score based on:
    - Total Transactions
    - Total Points Rewarded
    - Total Spending
    
    Parameters:
    - df: Pandas DataFrame containing transaction data.
    - top_n: Number of top stores to retrieve per MCC.
    - weights: Tuple of three values summing to 1, representing the weight for 
               (Transactions, Points Rewarded, Total Spending).
    
    Returns:
    - A DataFrame showing the top stores in each MCC based on the composite score.
    """

    # Grouping by MCC and Store Name, then calculating aggregates
    grouped = df.groupby(["MCC", "Store Name"]).agg(
        total_transactions=("TrxId", "count"),  # Count of transactions
        total_spending=("TotalPaid", "sum"),    # Total amount spent
        total_points=("PointsRewarded", "sum"), # Total points rewarded
        unique_users=("FK_BusinessUserId", "nunique")  # Unique customers
    ).reset_index()

    # Normalizing the values using MinMaxScaler (scales between 0 and 1)
    scaler = MinMaxScaler()
    grouped[["norm_transactions", "norm_spending", "norm_points"]] = scaler.fit_transform(
        grouped[["total_transactions", "total_spending", "total_points"]]
    )

    # Applying weighted sum formula
    w1, w2, w3 = weights
    grouped["composite_score"] = (
        w1 * grouped["norm_transactions"] +
        w2 * grouped["norm_points"] +
        w3 * grouped["norm_spending"]
    )

    # Sorting stores within each MCC based on the composite score
    top_stores = grouped.sort_values(
        by=["MCC", "composite_score"], ascending=[True, False]
    ).groupby("MCC").head(top_n)  # Keeping only the top N stores per MCC

    return top_stores

top_stores_in_each_mcc_df = get_top_stores_with_composite_score(df2, top_n=3, weights=(0.6, 0.2, 0.2))

top_stores_in_each_mcc_df = top_stores_in_each_mcc_df.merge(
    df2[["MCC", "MCC_Description"]].drop_duplicates(),
    on="MCC",
    how="left"
)

st.title("Top Stores by Composite Score in Each MCC")

st.caption("The composite score is calculated based on the total transactions, total points rewarded, and total spending in each store.")



mcc_mapping = dict(zip(top_stores_in_each_mcc_df["MCC_Description"], top_stores_in_each_mcc_df["MCC"]))

selected_mcc_desc = st.selectbox("Select MCC Category:", list(mcc_mapping.keys()) , key="top_stores_mcc")


selected_mcc = mcc_mapping[selected_mcc_desc]

filtered_store_data = top_stores_in_each_mcc_df[top_stores_in_each_mcc_df["MCC"] == selected_mcc]

if not filtered_store_data.empty:
    fig_pie = px.pie(
        filtered_store_data,
        names="Store Name",
        values="composite_score",
        title=f"Top Stores in MCC {selected_mcc} by Composite Score",
        hole=0.4,  # Optional: Creates a donut-style chart
        color_discrete_sequence=px.colors.qualitative.Set2  # Optional: Custom colors
    )

    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.warning("No data available for the selected MCC.")
    
    

def get_top_stores_by_unique_users(df, top_n=3):
    """
    Get exactly the top N stores in each MCC based on unique users.
    Resolves ties by breaking them using first appearance order.
    """
    # Grouping by MCC and Store Name, then counting unique users
    grouped = df.groupby(["MCC", "Store Name"]).agg(
        unique_users=("FK_BusinessUserId", "nunique")
    ).reset_index()

    # Sorting within each MCC by unique_users descending
    grouped = grouped.sort_values(by=["MCC", "unique_users"], ascending=[True, False])

    # Ranking stores within each MCC
    grouped["rank"] = grouped.groupby("MCC")["unique_users"].rank(method="first", ascending=False)

    # Keeping only the top N stores per MCC
    top_stores = grouped[grouped["rank"] <= top_n].drop(columns=["rank"])

    return top_stores

st.title("Top Stores by Unique Users in Each MCC")

st.caption("The number of unique users represents how many different customers interacted with each store.")

top_stores_unique_users = get_top_stores_by_unique_users(df2, top_n=2)


top_stores_unique_users = top_stores_unique_users.merge(
    df2[["MCC", "MCC_Description"]].drop_duplicates(),
    on="MCC",
    how="left"
)

mcc_mapping = dict(zip(top_stores_unique_users["MCC_Description"], top_stores_unique_users["MCC"]))

selected_mcc_desc = st.selectbox("Select MCC Category:", list(mcc_mapping.keys()) , key="top_stores_unique_users")


selected_mcc = mcc_mapping[selected_mcc_desc]

filtered_store_data = top_stores_unique_users[top_stores_unique_users["MCC"] == selected_mcc]

if not filtered_store_data.empty:
    fig_pie = px.pie(
        filtered_store_data,
        names="Store Name",
        values="unique_users",
        title=f"Top Stores in {selected_mcc_desc} by Unique Users",
        hole=0,  
        color_discrete_sequence=px.colors.qualitative.Set2  
    )

    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.warning("No data available for the selected MCC category.")




