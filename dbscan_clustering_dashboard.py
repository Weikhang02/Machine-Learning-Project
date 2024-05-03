import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import string
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define global stopwords list for sentiment analysis
stop_words = set(stopwords.words('english'))

def replace_empty_with_nan(df, column_names):
    for column in column_names:
        df[column] = df[column].apply(lambda x: np.nan if x == [] else x)
    return df

def clean_binary_columns(value):
    if value == 'Yes':
        return 1
    elif value == 'No':
        return 0
    else:
        return pd.NaT
    
def clean_rate_column(df, column_name):


    # Remove '/5' from the column values
    df[column_name] = df[column_name].str.replace('/5', '')

    # Convert the column to numeric, coercing errors to NaN
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    return df

def convert_to_numeric(df, column_name):
    # Remove commas from the column values
    df[column_name] = df[column_name].str.replace(',', '')

    # Convert the column to numeric, coercing errors to NaN
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    return df

def fillna_with_median(df, column_names):

    for column_name in column_names:
        median_value = df[column_name].median()
        df[column_name] = df[column_name].fillna(median_value)

    return df

def clean_text(text):
    # Remove non-alphanumeric characters
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def analyze_sentiment(review):
    review = clean_text(review)  # Clean the text
    testimonial = TextBlob(review)
    polarity = testimonial.sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"


def get_removal_condition(df, votes_threshold, rate_threshold, average_cost_threshold):
    removal_condition = (df['votes'] >= votes_threshold) | \
                        (df['rate'] <= rate_threshold) | \
                        (df['average_cost'] >= average_cost_threshold)

    return removal_condition

def clean_categorical_columns(value, common_values):
    if pd.isnull(value):
        return pd.NaT
    elif value in common_values:
        return value
    else:
        return 'Other'

def get_top_values(df, column_name, n=10):
    """
    Get the top n most common values in the column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    column_name (str): The name of the column to find the top values for.
    n (int): The number of top values to retrieve.

    Returns:
    list: A list of the top n values in the column.
    """
    return df[column_name].value_counts().index.tolist()[:n]

def clean_columns_using_top_values(df, columns):
    """
    Clean the specified columns in the DataFrame based on their top n values.

    Parameters:
    df (pandas.DataFrame): The DataFrame to operate on.
    columns (list of str): The names of the columns to clean.

    Returns:
    pandas.DataFrame: The DataFrame with cleaned categorical columns.
    """
    cleaned_df = df.copy()
    for column in columns:
        top_values = get_top_values(cleaned_df, column)
        cleaned_df[column] = cleaned_df[column].apply(clean_categorical_columns, common_values=top_values)
    return cleaned_df

def add_expensive_flag(df, cost_column, flag_column):
    """
    Add a binary column to the DataFrame indicating whether the cost is above the median.

    Parameters:
    df (pandas.DataFrame): The DataFrame to operate on.
    cost_column (str): The name of the column with cost values.
    flag_column (str): The name of the new binary column to be added.

    Returns:
    pandas.DataFrame: The DataFrame with the added binary column.
    """
    median_cost = df[cost_column].median()
    df[flag_column] = df[cost_column].apply(lambda x: 1 if x > median_cost else 0)
    df = df.drop(columns='average_cost')
    return df

def prepare_data_for_clustering(df, categorical_columns, numerical_columns):
    # One-Hot Encode Categorical Columns
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(df[categorical_columns])
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out())
    
    # Scale Numerical Columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Concatenate Encoded Categorical DataFrame and Scaled Numerical DataFrame
    final_df = pd.concat([df[numerical_columns].reset_index(drop=True),
                          encoded_categorical_df.reset_index(drop=True)], axis=1)
    
    return final_df

def preprocessing(df):
    df.rename(columns={'approx_cost(for two people)': 'average_cost', 'listed_in(city)': 'locality',}, inplace=True)    
    df = replace_empty_with_nan(df, ['menu_item', 'reviews_list'])
    df['online_order'] = df['online_order'].apply(clean_binary_columns)
    df['book_table'] = df['book_table'].apply(clean_binary_columns)
    df = clean_rate_column(df, 'rate')
    df = convert_to_numeric(df, 'average_cost')
    df = fillna_with_median(df, ['average_cost', 'rate'])
    df.dropna(subset=['rest_type', 'cuisines'], inplace=True)
    df.dropna(subset=['reviews_list'], inplace=True)
    df.rename(columns={'reviews_list': 'review_sentiment'}, inplace=True)
    stop_words = set(stopwords.words('english'))
    df['review_sentiment'] = df['review_sentiment'].apply(analyze_sentiment)
    df.drop('url', axis=1, inplace=True)
    df.drop('phone', axis=1, inplace=True)
    df.drop('address', axis=1, inplace=True)
    df.drop('name', axis=1, inplace=True)
    df.drop('location', axis=1, inplace=True)
    df.drop('dish_liked', axis=1, inplace=True)
    df.drop('menu_item',axis=1,inplace=True)
    votes_threshold = 6876
    rate_threshold = 2.4
    average_cost_threshold = 4500
    removal_condition = get_removal_condition(df, votes_threshold, rate_threshold, average_cost_threshold)
    df = df[~removal_condition]
    columns_to_clean = ['locality', 'rest_type', 'cuisines', 'listed_in(type)']
    df = clean_columns_using_top_values(df, columns_to_clean)
    df = add_expensive_flag(df, 'average_cost', 'expensive')
    categorical_columns = ['rest_type', 'cuisines', 'listed_in(type)', 'locality', 'review_sentiment']
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    final_df_for_clustering = prepare_data_for_clustering(df, categorical_columns, numerical_cols)
    return final_df_for_clustering

# Function to perform DBSCAN clustering
def cluster_data(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    return clusters

# Function to calculate silhouette score
def calculate_silhouette(data, clusters):
    noise = clusters == -1
    valid_clusters = clusters[~noise]
    valid_points = data[~noise]
    if len(np.unique(valid_clusters)) > 1:
        sil_score = silhouette_score(valid_points, valid_clusters)
        return sil_score
    else:
        return None

# Function to perform t-SNE and create a scatter plot
def plot_tsne(data, clusters):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    
    # Make sure to drop any NaN values from your data and clusters before fitting t-SNE
    data = data.dropna()
    clusters = clusters[~np.isnan(clusters)]
    
    tsne_results = tsne.fit_transform(data)
    
    # You can create a new figure here, if you are not in the middle of another plot
    fig, ax = plt.subplots()
    
    # Make sure the clusters array is a numpy array with the correct shape.
    # If clusters was a pandas series, converting to numpy array ensures compatibility.
    clusters = np.array(clusters)
    
    # Plot the t-SNE results with cluster labels
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    
    # Generate legend from scatter plot if there are no NaN values
    if not np.isnan(clusters).any():
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
    else:
        # Handle cases where the clusters contain NaN by not including a legend, or by setting a default legend
        print("NaN values in clusters, not displaying legend.")
    
    return fig


# Streamlit interface
st.title("DBSCAN Clustering Dashboard")

# Upload data file
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)

    # preprocessing
    data = preprocessing(data)
    
   # Inputs for DBSCAN
    eps = st.slider("Select eps", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    min_samples = st.slider("Select min_samples", min_value=1, max_value=50, value=5, step=1)

    # Button to perform clustering
    if st.button("Cluster"):
        with st.spinner('Clustering data...'):
            clusters = cluster_data(data, eps, min_samples)
            silhouette = calculate_silhouette(data, clusters)
            fig = plot_tsne(data, clusters)
            
            # Display results
            if silhouette is not None:
                st.success(f"Silhouette Score: {silhouette:.2f}")
            else:
                st.info("Silhouette score is not applicable due to the number of clusters.")
            st.pyplot(fig)
