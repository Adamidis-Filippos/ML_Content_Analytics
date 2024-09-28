import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification

# Define the paths
imdb_dataset_path = "Scraping and testing/imdb_movie_reviews.csv"
gpt2_model_directory = "GPT2"
distilbert_model_directory = "DistilBERT"

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv(imdb_dataset_path)
print(f"Dataset loaded. Number of reviews: {len(df)}")

# Load the trained GPT-2 model and tokenizer
print("Loading GPT-2 model and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_directory)
gpt2_model = GPT2ForSequenceClassification.from_pretrained(gpt2_model_directory)
gpt2_model.to(device)
gpt2_model.eval()
print("GPT-2 model and tokenizer loaded.")

# Load the trained DistilBERT model and tokenizer
print("Loading DistilBERT model and tokenizer...")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_directory)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_directory)
distilbert_model.to(device)
distilbert_model.eval()
print("DistilBERT model and tokenizer loaded.")

# Function to predict sentiment using GPT-2
def predict_sentiment_gpt2(text):
    inputs = gpt2_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()

    return predicted_class, confidence

# Function to predict sentiment using DistilBERT
def predict_sentiment_distilbert(text):
    inputs = distilbert_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = distilbert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()

    return predicted_class, confidence

# Predict sentiments for each review in the dataset
print("Starting sentiment predictions...")
results = []
for index, row in df.iterrows():
    review_text = row['Review']

    gpt2_label, gpt2_confidence = predict_sentiment_gpt2(review_text)
    distilbert_label, distilbert_confidence = predict_sentiment_distilbert(review_text)

    if gpt2_confidence > distilbert_confidence:
        final_label = "positive review" if gpt2_label == 1 else "negative review"
        final_confidence = gpt2_confidence
        model_used = "GPT-2"
    else:
        final_label = "positive review" if distilbert_label == 1 else "negative review"
        final_confidence = distilbert_confidence
        model_used = "DistilBERT"

    results.append({
        "Movie Title": row['Movie Title'],
        "Review": review_text,
        "Predicted Label": final_label,
        "Confidence": final_confidence,
        "Model Used": model_used
    })

    # Print progress every 100 reviews
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1}/{len(df)} reviews")

# Convert results to a DataFrame and save to a CSV file
print("Saving results to CSV file...")
results_df = pd.DataFrame(results)
results_df.to_csv("predicted_movie_reviews.csv", index=False)
print("Predictions saved to 'predicted_movie_reviews.csv'")
