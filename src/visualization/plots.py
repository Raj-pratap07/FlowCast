"""
FlowCast - Plotting Functions
Reusable plotting functions for Exploratory Data Analysis.
"""

import matplotlib.pyplot as plt 
import seaborn as sns 

#-----------------------
# Plot Style 
#------------------------
sns.set_style('whitegrid')

def plot_histogram(df, column, title, xlabel):
    # Plot histogram for a numerical column 

    plt.figure(figsize=(10,6))

    sns.histplot(
        data=df,
        x=column,
        bins=30,
        kde=True,
    )

    plt.title(title),
    plt.xlabel=(xlabel),
    plt.ylabel=("Freqency"),

    plt.tight_layout()
    plt.show()

def plot_bar_chart(df, column, title, xlabel):
    # Plot count plot for categorical data.

    plt.figure(figsize=(10,6))

    sns.countplot(
        data=df,
        x=column,
        order=df[column].value_counts().index
    )

    plt.title=(title)
    plt.xlabel=(xlabel)
    plt.ylabel=("Count")

    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()

def plot_box_plot(df, column, title):
    # Plot box plot to detect the outliers

    plt.figure(figsize=(10, 4))

    sns.boxplot(
        x=df[column]
    )

    plt.title(title)

    plt.tight_layout()
    plt.show()
