import os
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import ollama
import boto3
from dotenv import load_dotenv
from io import StringIO
from sklearn.preprocessing import MinMaxScaler

from app.core.Utilities import Utilities
from app.core.SqlServerService import SQLServerService

# Streamlit UI
st.set_page_config(page_title="MCC Score Dashboard", layout="wide")

# Loading environment variables
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

# SQL service initialization
sql_service = SQLServerService()


# Function to get URL parameters
def get_url_params():
    """Get URL parameters from the current page"""
    query_params = st.query_params
    return query_params


# Function to validate user ID
def validate_user_id(user_id):
    """Validate if user ID exists in database"""
    try:
        query = f"SELECT COUNT(*) as count FROM UserLoyaltyProfile WHERE FK_BusinessUserId = '{user_id}'"
        result = sql_service.query_to_df(query)
        return result["count"].iloc[0] > 0
    except Exception as e:
        st.error(f"Error validating user ID: {str(e)}")
        return False


# loading data
@st.cache_data
def load_data():
    df2 = sql_service.query_to_df("SELECT * FROM RewardTransactions", chunksize=100000)
    mcc_json = sql_service.query_to_df("SELECT * from MCC")
    df = Utilities.compute_mcc_interaction(df2, mcc_json)
    new_segments = Utilities.compute_user_features(df2)
    user_profiles = sql_service.query_to_df("SELECT * from UserLoyaltyProfile")
    return df, df2, new_segments, user_profiles


@st.cache_data
def load_user_specific_data(user_id):
    """Load data specific to a user"""
    try:
        user_transactions_query = f"""
        SELECT * FROM RewardTransactions 
        WHERE FkBusinessUserId = '{user_id}'
        """
        user_transactions = sql_service.query_to_df(user_transactions_query)

        user_profile_query = f"""
        SELECT * FROM UserLoyaltyProfile 
        WHERE FK_BusinessUserId = '{user_id}'
        """
        user_profile = sql_service.query_to_df(user_profile_query)

        return user_transactions, user_profile
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


# Get URL parameters
query_params = get_url_params()
url_user_id = query_params.get("user_id", None)

# Check if we're in single-user mode (URL parameter provided)
single_user_mode = url_user_id is not None

if single_user_mode:
    st.sidebar.info(f"🔍 **Single User Mode**\nUser ID: {url_user_id}")

    # Validate user ID
    if not validate_user_id(url_user_id):
        st.error(f"❌ User ID '{url_user_id}' not found in database!")
        st.stop()

    # Load user-specific data
    user_transactions, user_profile_data = load_user_specific_data(url_user_id)

    if user_transactions.empty or user_profile_data.empty:
        st.error(f"❌ No data found for User ID '{url_user_id}'")
        st.stop()

    # Show user-specific dashboard
    st.title(f"🧑‍💼 User Dashboard - ID: {url_user_id}")

    # User Profile Section
    user_data = user_profile_data.iloc[0]

    st.subheader("👤 User Profile Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Transactions", f"{user_data['TransactionCount']}")
        st.metric("Total Spent (KWD)", f"{user_data['TotalSpent']:.2f}")

    with col2:
        st.metric("Average Points/Trx", f"{user_data['AvgPointsPerTrx']:.2f}")
        st.metric("Average Amount/Trx", f"{user_data['AvgTransactionValue']:.2f}")

    with col3:
        st.metric("RFM Segment", user_data["RfmSegment"])
        st.metric("Recency (Days)", f"{user_data['RecencyDays']}")

    with col4:
        st.metric("Top MCC Category", user_data["TopMccCategory"])
        st.metric("Top Store", user_data["TopStore"])

    # Helper functions for single user mode
    def get_user_mcc_analysis(transactions_df):
        """Analyze user's MCC patterns"""
        if transactions_df.empty:
            return pd.DataFrame()

        mcc_analysis = (
            transactions_df.groupby(["Mcc", "MccDescription"])
            .agg(
                transaction_count=("TrxId", "count"),
                total_spent=("TotalPaid", "sum"),
                total_points=("PointsRewarded", "sum"),
                avg_spent_per_trx=("TotalPaid", "mean"),
            )
            .reset_index()
            .sort_values("total_spent", ascending=False)
        )
        return mcc_analysis

    def get_top_mccs_for_user(user_transactions, top_n=5, weights=(0.4, 0.3, 0.3)):
        """
        Computes the top MCC interests of a user based on frequency, total paid, and total points rewarded.

        Args:
            user_transactions (pd.DataFrame): DataFrame containing user's transaction data
            top_n (int): Number of top MCCs to return (default: 5)
            weights (tuple): Weights for (frequency, spending, points) in scoring (default: (0.4, 0.3, 0.3))

        Returns:
            pd.DataFrame: Top MCCs with interest scores and metrics
        """
        if user_transactions.empty:
            return pd.DataFrame()

        # Aggregating transactions per MCC and including MCC_Description
        mcc_aggregates = (
            user_transactions.groupby(["Mcc", "MccDescription"])
            .agg(
                transaction_count=("TrxId", "count"),
                total_spent=("TotalPaid", "sum"),
                total_points=("PointsRewarded", "sum"),
            )
            .reset_index()
        )

        # Add frequency column for normalization (same as transaction_count)
        mcc_aggregates["frequency"] = mcc_aggregates["transaction_count"]

        # Add avg_spent_per_trx for compatibility
        mcc_aggregates["avg_spent_per_trx"] = (
            mcc_aggregates["total_spent"] / mcc_aggregates["transaction_count"]
        )

        # Handle case where user has only one MCC (normalization would fail)
        if len(mcc_aggregates) == 1:
            mcc_aggregates["mcc_interest_score"] = 1.0
            return mcc_aggregates.head(top_n)

        # Normalizing values using MinMaxScaler
        scaler = MinMaxScaler()
        mcc_aggregates[["norm_frequency", "norm_spent", "norm_points"]] = (
            scaler.fit_transform(
                mcc_aggregates[["frequency", "total_spent", "total_points"]]
            )
        )

        # Computing the MCC interest score as a weighted combination
        w1, w2, w3 = weights
        mcc_aggregates["mcc_interest_score"] = (
            w1 * mcc_aggregates["norm_frequency"]
            + w2 * mcc_aggregates["norm_spent"]
            + w3 * mcc_aggregates["norm_points"]
        )

        # Sort and return the top MCC interests
        return mcc_aggregates.sort_values(
            by="mcc_interest_score", ascending=False
        ).head(top_n)

    def get_user_store_analysis(transactions_df):
        """Analyze user's store patterns"""
        if transactions_df.empty:
            return pd.DataFrame()

        store_analysis = (
            transactions_df.groupby("StoreName")
            .agg(
                transaction_count=("TrxId", "count"),
                total_spent=("TotalPaid", "sum"),
                total_points=("PointsRewarded", "sum"),
                avg_spent_per_trx=("TotalPaid", "mean"),
            )
            .reset_index()
            .sort_values("total_spent", ascending=False)
        )
        return store_analysis

    # User MCC Analysis
    st.subheader("📊 Top MCC Interests (MCC)")
    # user_mcc_data = get_user_mcc_analysis(user_transactions)
    user_mcc_data = get_top_mccs_for_user(user_transactions, top_n=10)

    if not user_mcc_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig_mcc_pie = px.pie(
                user_mcc_data.head(10),
                names="MccDescription",
                values="total_spent",
                title="Your Top MCC Interests",
            )
            st.plotly_chart(fig_mcc_pie, use_container_width=True)

        with col2:
            fig_mcc_bar = px.bar(
                user_mcc_data.head(10),
                x="transaction_count",
                y="MccDescription",
                orientation="h",
                title="Transaction Count by MCC",
            )
            st.plotly_chart(fig_mcc_bar, use_container_width=True)

        st.subheader("📋 Detailed MCC Analysis")
        st.dataframe(
            user_mcc_data[
                [
                    "MccDescription",
                    "transaction_count",
                    "total_spent",
                    "total_points",
                    "avg_spent_per_trx",
                ]
            ].round(2),
            use_container_width=True,
        )
    else:
        st.warning("No MCC data available for this user.")

    # User Store Analysis
    st.subheader("🏪 Your Favorite Stores (Merchants)")
    user_store_data = get_user_store_analysis(user_transactions)

    if not user_store_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig_store_pie = px.pie(
                user_store_data.head(10),
                names="StoreName",
                values="total_spent",
                title="Your Spending Distribution by Store (Merchants)",
            )
            st.plotly_chart(fig_store_pie, use_container_width=True)

        with col2:
            fig_store_bar = px.bar(
                user_store_data.head(10),
                x="transaction_count",
                y="StoreName",
                orientation="h",
                title="Transaction Count by Store (Merchants)",
            )
            st.plotly_chart(fig_store_bar, use_container_width=True)

        st.subheader("📋 Detailed Store (Merchants) Analysis")
        st.dataframe(
            user_store_data[
                [
                    "StoreName",
                    "transaction_count",
                    "total_spent",
                    "total_points",
                    "avg_spent_per_trx",
                ]
            ].round(2),
            use_container_width=True,
        )
    else:
        st.warning("No store data available for this user.")

    # Transaction Timeline
    st.subheader("📈 Your Transaction Timeline")
    if not user_transactions.empty:
        user_transactions["TrxDate"] = pd.to_datetime(user_transactions["TrxDate"])

        monthly_spending = (
            user_transactions.groupby(user_transactions["TrxDate"].dt.to_period("M"))
            .agg(total_spent=("TotalPaid", "sum"), transaction_count=("TrxId", "count"))
            .reset_index()
        )
        monthly_spending["TrxDate"] = monthly_spending["TrxDate"].astype(str)

        col1, col2 = st.columns(2)

        with col1:
            fig_timeline_spent = px.line(
                monthly_spending,
                x="TrxDate",
                y="total_spent",
                title="Monthly Spending Trend",
                markers=True,
            )
            st.plotly_chart(fig_timeline_spent, use_container_width=True)

        with col2:
            fig_timeline_count = px.line(
                monthly_spending,
                x="TrxDate",
                y="transaction_count",
                title="Monthly Transaction Count",
                markers=True,
            )
            st.plotly_chart(fig_timeline_count, use_container_width=True)

    # Add a back link to general dashboard
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 View General Dashboard"):
        st.query_params.clear()
        st.rerun()

else:
    # Original dashboard code for general view
    df, df2, new_segments, user_profiles = load_data()

    tab1, tab2 = st.tabs(["📊 MCC Score Analysis", "🧑‍💼 Users Profilings"])

    with tab1:
        st.title("📊 MCC Score Analysis Dashboard")

        # Sidebar Filters for MCC
        st.sidebar.header("Filters")

        # Getting Top 5 & Top 10 MCCs based on TotalPointsRewarded
        top_5_mccs = df.nlargest(5, "TotalPointsRewarded")["Mcc"].tolist()
        top_10_mccs = df.nlargest(10, "TotalPointsRewarded")["Mcc"].tolist()

        # Defining selection options
        all_mccs = df["Mcc"].unique().tolist()
        all_option = ["Select All", "Top 5", "Top 10"] + all_mccs

        # Multiselect with additional options
        selected_mccs = st.sidebar.multiselect(
            "Select MCC(s):", all_option, default=["Top 10"]
        )

        # Apply selection logic
        if "Select All" in selected_mccs:
            selected_mccs = all_mccs
        elif "Top 5" in selected_mccs:
            selected_mccs = top_5_mccs
        elif "Top 10" in selected_mccs:
            selected_mccs = top_10_mccs

        # Filtering the Data
        filtered_df = df[df["Mcc"].isin(selected_mccs)]

        # Displaying the Data Table
        st.subheader("MCC Score Data")
        st.dataframe(filtered_df)

        # Pie Chart for MCC Distribution
        st.subheader("MCC Distribution (Based on Total Points Rewarded)")
        fig_pie = px.pie(
            filtered_df,
            names="Mcc",
            values="TotalPointsRewarded",
            title="MCC Distribution by Total Points Rewarded",
            hole=0.4,
        )
        st.plotly_chart(fig_pie)

        # Scatter Plot plotting
        st.subheader("MCC Score vs Transaction Frequency")
        fig = px.scatter(
            filtered_df,
            x="TransactionFrequency",
            y="MCC_Score_Scaled",
            size="TotalPointsRewarded",
            color="Mcc",
            hover_data=["Mcc", "TotalPointsRewarded"],
            title="MCC Score vs Transaction Frequency",
        )
        st.plotly_chart(fig)

        # Bar Chart plotting
        st.subheader("Top MCCs by Score")
        fig_bar = px.bar(
            filtered_df,
            x="Mcc",
            y="MCC_Score_Scaled",
            color="MCC_Score_Scaled",
            title="Top MCCs by Score",
            labels={"Mcc": "MCC Category", "MCC_Score_Scaled": "MCC Score"},
        )
        st.plotly_chart(fig_bar)

        st.sidebar.markdown(
            "#### Note: The score is calculated with the following formula: MCC Score = Total Points Reward x Transaction Frequency / max(total Points Rewarded x Transaction Frequency)"
        )

        # Merge df2 with user_profiles to get cluster information
        df2_with_clusters = df2.merge(
            user_profiles[["FK_BusinessUserId", "ClusterName"]],
            left_on="FkBusinessUserId",
            right_on="FK_BusinessUserId",
            how="left",
        )

        # Now group by ClusterName and Mcc
        mcc_tier_grouped = (
            df2_with_clusters.groupby(["ClusterName", "Mcc"])
            .agg(
                TotalPointsRewarded=("PointsRewarded", "sum"),
                TransactionFrequency=("TrxId", "count"),
            )
            .reset_index()
        )

        mcc_tier_grouped = mcc_tier_grouped.dropna(subset=["ClusterName", "Mcc"])
        mcc_tier_grouped["ClusterName"] = mcc_tier_grouped["ClusterName"].astype(str)

        mcc_tier_grouped["MCC_Score"] = (
            mcc_tier_grouped["TotalPointsRewarded"]
            * mcc_tier_grouped["TransactionFrequency"]
        )
        mcc_tier_grouped["MCC_Score"] /= mcc_tier_grouped.groupby("ClusterName")[
            "MCC_Score"
        ].transform("max")
        mcc_tier_grouped = mcc_tier_grouped.dropna(subset=["MCC_Score"])

        top_mcc_per_tier = mcc_tier_grouped.loc[
            mcc_tier_grouped.groupby("ClusterName")["MCC_Score"].idxmax()
        ]

        st.subheader("📈 Users Segments")
        st.dataframe(new_segments)

        st.caption(
            "**Note:** The segments are based on five key features that describe user spending behavior as follows:\n\n **Total Amount Spent**, **Total Points Rewarded**, **Transaction Frequency**, **Average Spending Per Transaction**, **Average Points Per Transaction**"
        )

        st.subheader("🏆 Top MCC by Segments")
        st.dataframe(top_mcc_per_tier)

        st.subheader("🍰 Highest MCC Score Distribution by Segments")
        fig_pie = px.pie(
            top_mcc_per_tier,
            names="Mcc",
            values="MCC_Score",
            title="Top MCCs per Segments",
        )
        st.plotly_chart(fig_pie)

        st.sidebar.header("Segments Filters")
        tier_options = ["All"] + sorted(
            mcc_tier_grouped["ClusterName"].unique().tolist()
        )
        selected_tier = st.sidebar.selectbox("Select Segment:", tier_options)

        if selected_tier == "All":
            filtered_mcc_tier = mcc_tier_grouped
        else:
            filtered_mcc_tier = mcc_tier_grouped[
                mcc_tier_grouped["ClusterName"] == selected_tier
            ]

        st.subheader(
            f"🏆 Top MCCs for {selected_tier if selected_tier != 'All' else 'All Segments'}"
        )
        st.dataframe(
            filtered_mcc_tier.sort_values(by="MCC_Score", ascending=False).head(10)
        )

        st.subheader(
            f"📊 MCC Score Distribution for {selected_tier if selected_tier != 'All' else 'All Segments'}"
        )
        fig_bar = px.bar(
            filtered_mcc_tier.sort_values(by="MCC_Score", ascending=False).head(10),
            x="Mcc",
            y="MCC_Score",
            color="MCC_Score",
            title=f"Top MCCs by Score in {selected_tier if selected_tier != 'All' else 'All Segments'}",
            labels={"Mcc": "MCC Category", "MCC_Score": "MCC Score"},
        )
        st.plotly_chart(fig_bar)

        def get_top_stores_with_composite_score(
            df, top_n=3, weights=(1 / 3, 1 / 3, 1 / 3)
        ):
            grouped = (
                df.groupby(["Mcc", "StoreName"])
                .agg(
                    total_transactions=("TrxId", "count"),
                    total_spending=("TotalPaid", "sum"),
                    total_points=("PointsRewarded", "sum"),
                    unique_users=("FkBusinessUserId", "nunique"),
                )
                .reset_index()
            )

            scaler = MinMaxScaler()
            grouped[["norm_transactions", "norm_spending", "norm_points"]] = (
                scaler.fit_transform(
                    grouped[["total_transactions", "total_spending", "total_points"]]
                )
            )

            w1, w2, w3 = weights
            grouped["composite_score"] = (
                w1 * grouped["norm_transactions"]
                + w2 * grouped["norm_points"]
                + w3 * grouped["norm_spending"]
            )

            top_stores = (
                grouped.sort_values(
                    by=["Mcc", "composite_score"], ascending=[True, False]
                )
                .groupby("Mcc")
                .head(top_n)
            )

            return top_stores

        top_stores_in_each_mcc_df = get_top_stores_with_composite_score(
            df2_with_clusters, top_n=3, weights=(0.6, 0.2, 0.2)
        )

        top_stores_in_each_mcc_df = top_stores_in_each_mcc_df.merge(
            df2_with_clusters[["Mcc", "MccDescription"]].drop_duplicates(),
            on="Mcc",
            how="left",
        )

        st.title("Top Stores by Composite Score in Each MCC")

        st.caption(
            "The composite score is calculated based on the total transactions, total points rewarded, and total spending in each store."
        )

        mcc_mapping = dict(
            zip(
                top_stores_in_each_mcc_df["MccDescription"],
                top_stores_in_each_mcc_df["Mcc"],
            )
        )

        selected_mcc_desc = st.selectbox(
            "Select MCC Category:", list(mcc_mapping.keys()), key="top_stores_mcc"
        )

        selected_mcc = mcc_mapping[selected_mcc_desc]

        filtered_store_data = top_stores_in_each_mcc_df[
            top_stores_in_each_mcc_df["Mcc"] == selected_mcc
        ]

        if not filtered_store_data.empty:
            fig_pie = px.pie(
                filtered_store_data,
                names="StoreName",
                values="composite_score",
                title=f"Top Stores in MCC {selected_mcc} by Composite Score",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No data available for the selected MCC.")

    with tab2:
        st.title("🧑‍💼 Users Profilings Dashboard")

        # Add option to view specific user
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Quick User Lookup")

        # Input for user ID
        quick_user_id = st.sidebar.text_input("Enter User ID for detailed view:")
        if st.sidebar.button("View User Details") and quick_user_id:
            if validate_user_id(quick_user_id):
                st.query_params["user_id"] = quick_user_id
                st.rerun()
            else:
                st.sidebar.error("User ID not found!")

        # Dropdown to select a user
        user_ids = user_profiles["FK_BusinessUserId"].unique()
        selected_user = st.selectbox("Select a User ID:", user_ids)

        user_data = user_profiles[
            user_profiles["FK_BusinessUserId"] == selected_user
        ].iloc[0]

        st.subheader(f"User Profile for User ID: {selected_user}")

        # Add button to view detailed dashboard for this user
        if st.button(f"🔍 View Detailed Dashboard for User {selected_user}"):
            st.query_params["user_id"] = selected_user
            st.rerun()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", f"{user_data['TransactionCount']}")
        col2.metric("Average Points Rewarded", f"{user_data['AvgPointsPerTrx']:.2f}")
        col3.metric("Most Active Day", user_data["BestDayByTrx"])

        col1.metric("Total Amount Spent (KWD)", f"{user_data['TotalSpent']:.2f}")
        col2.metric(
            "Average Amount Spent (KWD)", f"{user_data['AvgTransactionValue']:.2f}"
        )
        col3.metric("Most Active Month", user_data["BestMonthByTrx"])

        col1.metric(
            "Recency (Days since last transaction)", f"{user_data['RecencyDays']}"
        )
        col2.metric(
            "Most Common Season",
            (
                "Spring"
                if user_data["BestMonthByTrx"] in ["March", "April", "May"]
                else (
                    "Summer"
                    if user_data["BestMonthByTrx"] in ["June", "July", "August"]
                    else (
                        "Autumn"
                        if user_data["BestMonthByTrx"]
                        in ["September", "October", "November"]
                        else "Winter"
                    )
                )
            ),
        )

        col3.metric("Spending Growth Rate", f"{user_data['SpendingGrowthRate']:.2%}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Best MCC Category", user_data["TopMccCategory"])
        col2.metric("Best Store", user_data["TopStore"])
        col3.metric("RFM Segment", user_data["RfmSegment"])

        # Summary Metrics
        st.subheader("📌 Users Profile Summary")
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Users", f"{user_profiles.shape[0]:,}")
        col2.metric(
            "Avg Transaction Count", f"{user_profiles['TransactionCount'].mean():.2f}"
        )
        col3.metric("Avg Total Spending", f"${user_profiles['TotalSpent'].mean():,.2f}")

        st.subheader("🛠️ User RFM")
        fig_segment = px.pie(
            user_profiles,
            names="RfmSegment",
            title="User Distribution by RFM Segment",
            hole=0.4,
        )
        st.plotly_chart(fig_segment, use_container_width=True)

# # Add information about URL parameters in sidebar
# st.sidebar.markdown("---")
# st.sidebar.markdown("### 🔗 External App Integration")
# st.sidebar.markdown(
#     """
# **For External Applications:**

# Simply add the user ID as a URL parameter:

# `http://your-streamlit-url:8501?user_id=USER_ID`

# **Example:**
# `http://your-streamlit-url:8501?user_id=12345`

# This will show a personalized dashboard for that specific user.
# """
# )
