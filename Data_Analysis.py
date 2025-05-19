import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Netflix TV Shows and Movies.csv')
df.info()
df.head()

# Data Cleaning 
print(df.info())
print(df.isnull().sum())
df.drop_duplicates(inplace=True)
df.dropna(subset=['description', 'imdb_score'], inplace=True)
df['age_certification'].fillna('Unknown', inplace=True)
df['imdb_votes'] = pd.to_numeric(df['imdb_votes'], errors='coerce')
df = df[df['runtime'] > 0]
df['title'] = df['title'].str.strip()
df['type'] = df['type'].str.upper()
df = df[(df['release_year'] >= 1900) & (df['release_year'] <= 2025)]
df.to_csv("Netflix_Cleaned.csv", index=False)

# Set visual style
sns.set(style="whitegrid")

# Basic summary statistics
summary = df.describe(include='all')

# Distribution of content types (Movies vs Shows)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='type', palette='Set2')
plt.title('Distribution of Content Types on Netflix')
plt.xlabel('Content Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Distribution of release years
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='release_year', bins=30, kde=True, color='skyblue')
plt.title('Distribution of Release Years')
plt.xlabel('Release Year')
plt.ylabel('Number of Titles')
plt.tight_layout()
plt.show()

# Age Certification Breakdown
# Bar plot for Age Certification
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='age_certification', order=df['age_certification'].value_counts().index, palette='coolwarm')
plt.title('Age Certification Breakdown on Netflix')
plt.xlabel('Age Certification')
plt.ylabel('Number of Titles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Pie chart for Age Certification distribution
plt.figure(figsize=(8, 8))
df['age_certification'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Age Certification Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# IMDB Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['imdb_score'], bins=30, kde=True, color='salmon')
plt.title('Distribution of IMDB Scores')
plt.xlabel('IMDB Score')
plt.ylabel('Number of Titles')
plt.tight_layout()
plt.show()

# Runtime distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['runtime'], bins=50, kde=True, color='lightgreen')
plt.title('Distribution of Runtime')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Number of Titles')
plt.tight_layout()
plt.show()

# IMDB score vs votes
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='imdb_votes', y='imdb_score', hue='type', alpha=0.6)
plt.title('IMDB Score vs. IMDB Votes')
plt.xlabel('IMDB Votes')
plt.ylabel('IMDB Score')
plt.legend(title='Type')
plt.tight_layout()
plt.show()

#Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()

# Time Trend / Yearly IMDb Average
avg_score_by_year = df.groupby('release_year')['imdb_score'].mean().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(data=avg_score_by_year, x='release_year', y='imdb_score')
plt.title("Average IMDb Score by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Average IMDb Score")
plt.tight_layout()
plt.show()

# Top rated Titles
top_rated = df[df['imdb_votes'] > 1000]
top_rated = top_rated.sort_values(by='imdb_score', ascending=False)
top_10 = top_rated[['title', 'imdb_score', 'imdb_votes', 'type', 'release_year']].drop_duplicates().head(10)
top_10 = top_10.sort_values(by='imdb_score')
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10, x='imdb_score', y='title', palette='viridis', orient='h')
plt.title('Top 10 Highest Rated Titles on Netflix (IMDb Score, min 1000 votes)', fontsize=14)
plt.xlabel('IMDb Score')
plt.ylabel('Title')
plt.tight_layout()
plt.show()
print("\nTop 10 Highest Rated Titles on Netflix:")
print(top_10)