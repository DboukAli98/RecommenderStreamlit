import os
from pathlib import Path
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import ollama



# Streamlit UI
st.set_page_config(page_title="MCC Score Dashboard", layout="wide")

#setting dynamic directories

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
base_dir = parent_dir.parent

data_dir = base_dir / "data"


#loading data 

@st.cache_data
def load_data():
    df = pd.read_csv(data_dir / "mcc_scores.csv")
    df = df.drop(columns=["Unnamed: 0"])
    df2 = pd.read_csv(data_dir /  "Cleaned/rewards_transactions_cleaned.csv")
    new_segments = pd.read_csv(data_dir / "Cleaned/new_segments.csv")
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

if st.button("✨" , help="Click to generate an AI-powered summary using **Related's** AI engine for MCC Scores"):
    with st.spinner("Analyzing data..."):
        
        df_string = filtered_df.to_string()

        response_placeholder = st.empty()

        response_text = ""
        for chunk in ollama.chat(
            model="llama3.2:latest",
            messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a TAM Rewards loyalty program analyst for KFH Bank in Kuwait. "
                            "Your goal is to analyze user spending trends based on reward transactions. "
                            "Provide insights on user purchasing behavior, which categories dominate, and "
                            "how to encourage spending in underutilized categories."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The data below represents what users **buy** using their credit cards in the TAM Rewards program. "
                            f"Analyze the spending behavior, highlight the most popular spending categories, and suggest "
                            f"ways to encourage spending in lower MCC categories:\n\n{df_string}"
                        )
                    }
                ],
            stream=True
        ):
            response_text += chunk["message"]["content"] 
            response_placeholder.markdown(response_text)

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




