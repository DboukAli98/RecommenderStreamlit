# imports
from bs4 import BeautifulSoup
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


class Utilities:
    """
    Utilities class for utility functions.
    """

    @staticmethod
    def check_nan_values(df, return_dict=False):
        """
        Function to check null (NaN) values in each column of the DataFrame.
        If return_dict=True, returns a dictionary instead of printing.
        """
        null_counts = df.isnull().sum().to_dict()
        if return_dict:
            return null_counts
        else:
            for col, count in null_counts.items():
                print(f"{col}: {count} null (NaN) values")

    @staticmethod
    def remove_html_tags(text):
        """Function to remove html tags from text"""
        return BeautifulSoup(text, "html.parser").get_text()

    @staticmethod
    def compute_mcc_interaction(rewards_df, mccs_json):
        """Function that computes MCC interaction"""
        # Fixed: Use correct column names from database
        mcc_interaction = (
            rewards_df.groupby("Mcc")  # Changed from "MCC" to "Mcc"
            .agg(
                TotalPointsRewarded=("PointsRewarded", "sum"),
                TransactionFrequency=("TrxId", "count"),
            )
            .reset_index()
        )

        mcc_interaction["MCC_Score"] = (
            mcc_interaction["TotalPointsRewarded"]
            * mcc_interaction["TransactionFrequency"]
        ) / (
            mcc_interaction["TotalPointsRewarded"]
            * mcc_interaction["TransactionFrequency"]
        ).max()

        mcc_interaction_sorted = mcc_interaction.sort_values(
            by="MCC_Score", ascending=False
        )

        mcc_interaction_sorted["MCC_Score_Log"] = np.log1p(
            mcc_interaction_sorted["MCC_Score"]
        )

        mcc_interaction_sorted["MCC_Score_Scaled"] = (
            mcc_interaction_sorted["MCC_Score"]
            - mcc_interaction_sorted["MCC_Score"].min()
        ) / (
            mcc_interaction_sorted["MCC_Score"].max()
            - mcc_interaction_sorted["MCC_Score"].min()
        )

        # Ensure data types match for merge
        mcc_interaction_sorted["Mcc"] = mcc_interaction_sorted["Mcc"].astype(str)

        # Prepare mccs_json for merge - ensure consistent column names and data types
        mccs_json_prepared = mccs_json.copy()
        mccs_json_prepared["MCC"] = mccs_json_prepared["MCC"].astype(str)
        mccs_json_prepared = mccs_json_prepared.rename(columns={"MCC": "Mcc"})

        # Merge with MCC descriptions
        mcc_interaction_sorted = mcc_interaction_sorted.merge(
            mccs_json_prepared, on="Mcc", how="left"
        )

        return mcc_interaction_sorted

    @staticmethod
    def compute_user_features(rewards_df):
        """Function that computes User Features based K-means clustering"""
        # Fixed: Use correct column name "FkBusinessUserId" instead of "FK_BusinessUserId"
        user_features = (
            rewards_df.groupby("FkBusinessUserId")
            .agg(
                Total_Spent=("TotalPaid", "sum"),
                Total_Points=("PointsRewarded", "sum"),
                Transaction_Count=("TrxId", "count"),
                Avg_Spending_Per_Transaction=("TotalPaid", "mean"),
                Avg_Points_Per_Transaction=("PointsRewarded", "mean"),
            )
            .reset_index()
        )

        # Normalizing Features for Clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(
            user_features[
                [
                    "Total_Spent",
                    "Total_Points",
                    "Transaction_Count",
                    "Avg_Spending_Per_Transaction",
                    "Avg_Points_Per_Transaction",
                ]
            ]
        )

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(features_scaled)

        cluster_mapping = {
            0: "Low Spender",
            1: "Very High Spender",
            2: "Medium Spender",
            3: "High Spender",
        }
        user_features["Cluster"] = kmeans_labels
        user_features["Cluster_Name"] = user_features["Cluster"].map(cluster_mapping)

        cluster_validation = (
            user_features.groupby("Cluster_Name")
            .agg(
                Avg_Total_Spent=("Total_Spent", "mean"),
                Avg_Total_Points=("Total_Points", "mean"),
                Avg_Transaction_Count=("Transaction_Count", "mean"),
                Users_Per_Cluster=("Cluster", "count"),
            )
            .sort_values(by="Avg_Total_Spent", ascending=False)
        )

        return cluster_validation
