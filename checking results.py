import pandas as pd

results_df = pd.read_csv("predicted_movie_reviews.csv")

import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV into a DataFrame
df = pd.read_csv("predicted_movie_reviews.csv")

# Count the number of positive and negative reviews for each movie
review_counts = df.groupby(['Movie Title', 'Predicted Label']).size().unstack(fill_value=0)

# Plotting the number of positive and negative reviews per movie
review_counts.plot(kind='bar', stacked=True, figsize=(12, 8), color=['red', 'green'])
plt.title('Number of Positive and Negative Reviews per Movie')
plt.xlabel('Movie Title')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=90)
plt.legend(title='Review Sentiment')
plt.tight_layout()
plt.show()

# Calculate the average confidence for each movie and label
confidence_scores = df.groupby(['Movie Title', 'Predicted Label'])['Confidence'].mean().unstack()

# Plotting the average confidence scores for positive and negative reviews per movie
confidence_scores.plot(kind='bar', figsize=(12, 8), color=['red', 'green'])
plt.title('Average Confidence Scores for Positive and Negative Reviews per Movie')
plt.xlabel('Movie Title')
plt.ylabel('Average Confidence')
plt.xticks(rotation=90)
plt.legend(title='Review Sentiment')
plt.tight_layout()
plt.show()

# Optionally, visualize the distribution of confidence scores
plt.figure(figsize=(12, 8))
sns.boxplot(x='Movie Title', y='Confidence', hue='Predicted Label', data=df, palette=['red', 'green'])
plt.title('Distribution of Confidence Scores by Movie and Sentiment')
plt.xlabel('Movie Title')
plt.ylabel('Confidence')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()




# Group the data by Movie Title and Predicted Label, then count the occurrences
movie_label_counts = results_df.groupby(['Movie Title', 'Predicted Label']).size().unstack(fill_value=0)

# Plotting the number of positive and negative reviews per movie
plt.figure(figsize=(12, 8))
movie_label_counts.plot(kind='bar', stacked=True, color=['red', 'green'])
plt.title('Distribution of Positive and Negative Reviews per Movie')
plt.xlabel('Movie Title')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=90)
plt.legend(title='Predicted Label')
plt.tight_layout()

# Show the plot
plt.show()
