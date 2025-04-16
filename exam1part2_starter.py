import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Automobile Dataset Analysis")

# 1. Load Data
st.subheader("1. Load Dataset")
url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/main/CleanedAutomobile.csv'
df = pd.read_csv(url)
st.write("Preview of the dataset:")
st.dataframe(df.head())

# 2. Data Types
st.subheader("2. Data Types")
st.write(df.dtypes)

# 3. Correlation Analysis
st.subheader("3. Correlation: bore, stroke, compression-ratio, and horsepower")
corr = df[["bore", "stroke", "compression-ratio", "horsepower"]].corr()
st.dataframe(corr)

# 4. Scatterplots with Regression Line
st.subheader("4. Scatterplots with Regression Line")

def plot_reg(x, y):
    fig, ax = plt.subplots()
    sns.regplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(f"{y} vs {x}")
    st.pyplot(fig)

plot_reg("engine-size", "price")
plot_reg("highway-mpg", "price")
plot_reg("peak-rpm", "price")
plot_reg("stroke", "price")

st.write("Correlation between stroke and price:")
st.write(df[["stroke", "price"]].corr())

# 5. Boxplots for Categorical Variables
st.subheader("5. Boxplots: Categorical Variables vs Price")

def plot_box(x):
    fig, ax = plt.subplots()
    sns.boxplot(x=x, y="price", data=df, ax=ax)
    ax.set_title(f"Price by {x}")
    st.pyplot(fig)

plot_box("body-style")
plot_box("engine-location")
plot_box("drive-wheels")

# 6. Descriptive Stats
st.subheader("6. Descriptive Statistics")
st.write(df.describe())
st.write("Object Type Summary:")
st.write(df.describe(include=['object']))

# 7. Value Counts
st.subheader("7. Value Counts for drive-wheels and engine-location")

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.columns = ['value_counts']
drive_wheels_counts.index.name = 'drive-wheels'
st.dataframe(drive_wheels_counts)

engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.columns = ['value_counts']
engine_loc_counts.index.name = 'engine-location'
st.dataframe(engine_loc_counts)

# 8. Grouping and Pivot
st.subheader("8. Grouping and Pivot Tables")
df_group = df[['drive-wheels','body-style','price']]
grouped = df_group.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
pivot = grouped.pivot(index='drive-wheels', columns='body-style', values='price')
pivot = pivot.fillna(0)
st.dataframe(pivot)

# Heatmap
st.subheader("Heatmap: Drive-wheels and Body-style vs Price")
fig, ax = plt.subplots()
im = ax.pcolor(pivot, cmap='RdBu')
ax.set_xticks(np.arange(pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(pivot.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(pivot.columns, rotation=90)
ax.set_yticklabels(pivot.index)
fig.colorbar(im)
st.pyplot(fig)

# Average price by body-style
st.subheader("9. Average Price by Body Style")
avg_price_by_body = df.groupby('body-style')['price'].mean()
st.write(avg_price_by_body)
