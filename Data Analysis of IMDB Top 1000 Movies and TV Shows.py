# -*- coding: utf-8 -*-
"""ZAIO_Assignment_1_IMDB_Analysis.ipynb

IMDB Top 1000 Movies Analysis Assignment
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the local path to the dataset
file_path = r"C:\Users\hp\OneDrive\Desktop\Masters\ZAIO\Assignment\imdb-dataset\imdb_top_1000.csv"

# Load the dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully.")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

df = load_dataset(file_path)


# Only proceed if dataset was loaded successfully
if df is not None:
    # Display basic info
    print("\nDataset Info:")
    print(df.info())

    # Display first 5 rows
    print("\nFirst 5 rows:")
    print(df.head())
else:
    print("Cannot proceed with analysis - dataset not loaded.")
    exit()

"""## **Phase 2: Data Preparation**"""

def clean_data(df):
    """Perform all data cleaning operations"""
    try:
        # Check for missing values
        print("\nMissing values per column:")
        print(df.isnull().sum())

        # Handle missing values
        # For numerical columns, fill with median
        num_cols = ['Meta_score', 'No_of_Votes', 'Gross']
        for col in num_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

        # For categorical columns, fill with mode or 'Unknown'
        cat_cols = ['Certificate', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']
        for col in cat_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna('Unknown', inplace=True)

        # Check for duplicates
        print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

        # Feature Engineering
        # Convert Runtime to numeric
        if 'Runtime' in df.columns:
            df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(int)

        # Convert Gross to numeric (remove commas and dollar signs)
        if 'Gross' in df.columns:
            df['Gross'] = df['Gross'].str.replace(',', '').str.replace('$', '').astype(float)

        # Create Decade column
        if 'Released_Year' in df.columns:
            df['Decade'] = (df['Released_Year'].astype(int) // 10 * 10).astype(str) + 's'

        # Create Lead Actors column
        star_cols = ['Star1', 'Star2', 'Star3', 'Star4']
        if all(col in df.columns for col in star_cols):
            df['Lead_Actors'] = df[star_cols].apply(
                lambda x: ', '.join(x.dropna().astype(str)), axis=1)

        # Display cleaned dataset info
        print("\nCleaned Dataset Info:")
        print(df.info())
        
        return df
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

# Clean the data
df_clean = clean_data(df.copy())

if df_clean is None:
    print("Data cleaning failed. Using original dataset.")
    df_clean = df

"""## **Phase 3: Data Visualization**"""

def create_visualizations(df):
    """Generate all required visualizations"""
    try:
        # Visualization 1: Distribution of IMDB Ratings vs. Meta Scores
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='IMDB_Rating', y='Meta_score', data=df, alpha=0.6)
        plt.title('IMDB Rating vs. Meta Score')
        plt.xlabel('IMDB Rating')
        plt.ylabel('Meta Score')
        plt.show()

        # Visualization 2: Top 10 Genres by Frequency
        plt.figure(figsize=(12, 6))
        top_genres = df['Genre'].str.split(', ').explode().value_counts().head(10)
        sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
        plt.title('Top 10 Movie Genres')
        plt.xlabel('Count')
        plt.ylabel('Genre')
        plt.show()

        # Visualization 3: Gross vs. Number of Votes
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='No_of_Votes', y='Gross', data=df, alpha=0.6)
        plt.title('Gross Revenue vs. Number of Votes')
        plt.xlabel('Number of Votes')
        plt.ylabel('Gross Revenue ($)')
        plt.show()

        # Visualization 4: IMDB Rating by Certificate
        plt.figure(figsize=(12, 6))
        cert_order = ['U', 'PG', 'PG-13', 'R', 'A', 'UA']
        available_certs = [cert for cert in cert_order if cert in df['Certificate'].unique()]
        sns.boxplot(x='Certificate', y='IMDB_Rating', data=df, order=available_certs)
        plt.title('IMDB Rating Distribution by Certificate')
        plt.xlabel('Certificate')
        plt.ylabel('IMDB Rating')
        plt.xticks(rotation=45)
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        return False

# Create visualizations
if not create_visualizations(df_clean):
    print("Some visualizations failed to generate")

"""## **Phase 4: Applied Statistical Analysis**"""

def perform_statistical_analysis(df):
    """Perform statistical analysis"""
    try:
        # Descriptive Statistics
        print("\nDescriptive Statistics for Numerical Columns:")
        num_cols = ['IMDB_Rating', 'Meta_score', 'Runtime', 'No_of_Votes', 'Gross']
        available_num_cols = [col for col in num_cols if col in df.columns]
        print(df[available_num_cols].describe())

        # Correlation Analysis
        print("\nCorrelation Matrix:")
        print(df[available_num_cols].corr())

        # Outlier Detection using IQR for Gross
        if 'Gross' in df.columns:
            Q1 = df['Gross'].quantile(0.25)
            Q3 = df['Gross'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df['Gross'] < lower_bound) | (df['Gross'] > upper_bound)]
            print(f"\nNumber of outliers in Gross: {len(outliers)}")
            
        return True
        
    except Exception as e:
        print(f"Error during statistical analysis: {str(e)}")
        return False

# Perform statistical analysis
if not perform_statistical_analysis(df_clean):
    print("Statistical analysis encountered errors")

"""## **Phase 5: Advanced Analysis**"""

def advanced_analysis(df):
    """Perform advanced analysis"""
    try:
        # Analysis 1: Top Directors by Average Gross
        if 'Director' in df.columns and 'Gross' in df.columns:
            top_directors = df.groupby('Director')['Gross'].mean().sort_values(ascending=False).head(5)
            print("\nTop 5 Directors by Average Gross Revenue:")
            print(top_directors)

            plt.figure(figsize=(12, 6))
            top_directors.plot(kind='bar', color='skyblue')
            plt.title('Top 5 Directors by Average Gross Revenue')
            plt.xlabel('Director')
            plt.ylabel('Average Gross Revenue ($ millions)')
            plt.xticks(rotation=45)
            plt.show()

        # Analysis 2: Most Frequent Actors in Top-Rated Movies
        if 'Star1' in df.columns and 'IMDB_Rating' in df.columns:
            top_actors = df[df['IMDB_Rating'] > 8.5]['Star1'].value_counts().head(5)
            print("\nMost Frequent Lead Actors in Top-Rated Movies (IMDB > 8.5):")
            print(top_actors)

        # Analysis 3: Actor Pairs with Highest Gross
        if all(col in df.columns for col in ['Star1', 'Star2', 'Gross']):
            df['Actor_Pair'] = df['Star1'] + ' & ' + df['Star2']
            top_pairs = df.groupby('Actor_Pair')['Gross'].mean().sort_values(ascending=False).head(5)
            print("\nTop 5 Actor Pairs by Average Gross Revenue:")
            print(top_pairs)

        # Analysis 4: Genre vs. IMDB Rating
        if 'Genre' in df.columns and 'IMDB_Rating' in df.columns:
            genre_ratings = df.copy()
            genre_ratings['Genre'] = genre_ratings['Genre'].str.split(', ')
            genre_ratings = genre_ratings.explode('Genre')

            plt.figure(figsize=(14, 8))
            top_genres = genre_ratings['Genre'].value_counts().index[:10]
            sns.boxplot(x='Genre', y='IMDB_Rating', data=genre_ratings[genre_ratings['Genre'].isin(top_genres)])
            plt.title('IMDB Rating Distribution by Genre (Top 10)')
            plt.xlabel('Genre')
            plt.ylabel('IMDB Rating')
            plt.xticks(rotation=45)
            plt.show()
            
        return True
        
    except Exception as e:
        print(f"Error during advanced analysis: {str(e)}")
        return False

# Perform advanced analysis
if not advanced_analysis(df_clean):
    print("Advanced analysis encountered errors")

"""## **Phase 6: Conclusion & Report Preparation**"""

def generate_summary(df):
    """Generate summary statistics for report"""
    try:
        summary_data = {
            'Metric': ['Total Movies', 'Average IMDB Rating', 'Average Gross Revenue', 
                       'Most Common Genre', 'Top Director by Gross', 'Top Actor in High-Rated Movies'],
            'Value': []
        }

        # Total Movies
        summary_data['Value'].append(len(df))

        # Average IMDB Rating
        if 'IMDB_Rating' in df.columns:
            summary_data['Value'].append(round(df['IMDB_Rating'].mean(), 2))
        else:
            summary_data['Value'].append('N/A')

        # Average Gross Revenue
        if 'Gross' in df.columns:
            summary_data['Value'].append(f"${round(df['Gross'].mean(), 2)}")
        else:
            summary_data['Value'].append('N/A')

        # Most Common Genre
        if 'Genre' in df.columns:
            summary_data['Value'].append(df['Genre'].str.split(', ').explode().mode()[0])
        else:
            summary_data['Value'].append('N/A')

        # Top Director by Gross
        if 'Director' in df.columns and 'Gross' in df.columns:
            top_dir = df.groupby('Director')['Gross'].mean().idxmax()
            summary_data['Value'].append(top_dir)
        else:
            summary_data['Value'].append('N/A')

        # Top Actor in High-Rated Movies
        if 'Star1' in df.columns and 'IMDB_Rating' in df.columns:
            top_actor = df[df['IMDB_Rating'] > 8.5]['Star1'].value_counts().idxmax()
            summary_data['Value'].append(top_actor)
        else:
            summary_data['Value'].append('N/A')

        summary_df = pd.DataFrame(summary_data)
        print("\nKey Findings Summary:")
        print(summary_df)
        
        return summary_df
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return None

# Generate summary
summary = generate_summary(df_clean)

# Save cleaned dataset
try:
    df_clean.to_csv('cleaned_imdb_top_1000.csv', index=False)
    print("\nCleaned dataset saved as 'cleaned_imdb_top_1000.csv'")
except Exception as e:
    print(f"\nError saving cleaned dataset: {str(e)}")

print("\nAnalysis complete!")