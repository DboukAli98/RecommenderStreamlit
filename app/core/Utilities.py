#imports
from bs4 import BeautifulSoup
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
        """Function to remove htl tags from text"""
        return BeautifulSoup(text , "html.parser").get_text()