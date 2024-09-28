import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import tkinter as tk
from tkinter import messagebox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained GPT-2 model and tokenizer
gpt2_model_directory = "GPT2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_directory)
gpt2_model = GPT2ForSequenceClassification.from_pretrained(gpt2_model_directory)
gpt2_model.to(device)
gpt2_model.eval()

# Load the trained DistilBERT model and tokenizer
distilbert_model_directory = "DistilBERT"
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_directory)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_directory)
distilbert_model.to(device)
distilbert_model.eval()

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

# Function to handle the button click
def on_predict():
    input_text = entry.get()
    if input_text.lower() == 'exit':
        root.destroy()
        return

    gpt2_label, gpt2_confidence = predict_sentiment_gpt2(input_text)
    distilbert_label, distilbert_confidence = predict_sentiment_distilbert(input_text)

    if gpt2_confidence > distilbert_confidence:
        final_label = "positive review" if gpt2_label == 1 else "negative review"
        final_confidence = gpt2_confidence
        model_used = "GPT-2"
    else:
        final_label = "positive review" if distilbert_label == 1 else "negative review"
        final_confidence = distilbert_confidence
        model_used = "DistilBERT"

    messagebox.showinfo("Prediction Result", f"Predicted Label: {final_label}\nConfidence: {final_confidence:.2f}\nModel: {model_used}")

# Create the main window
root = tk.Tk()
root.title("Sentiment Analysis")

# Create a label
label = tk.Label(root, text="Enter a review:")
label.pack(pady=10)

# Create an entry box
entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Create a button
button = tk.Button(root, text="Predict Sentiment", command=on_predict)
button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()