import json
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from chatbot.preprocess import preprocess_text  # Ensure this exists

def load_intents(file_path):
    """Load intents from a JSON file."""
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            intents = json.load(f)
        return intents
    except FileNotFoundError:
        print(f"❌ Error: Intents file '{file_path}' not found.")
        exit(1)

def prepare_training_data(intents):
    """Prepare training data from intents."""
    questions = []
    tags = []
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            questions.append(pattern)
            tags.append(intent['tag'])
    
    return pd.DataFrame({'question': questions, 'intent': tags})

def train_model():
    # Define paths
    intents_path = "chatbot/data/intents.json"
    model_path = "chatbot/data/model.pkl"

    # Ensure 'chatbot/data/' exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load intents from the file
    intents = load_intents(intents_path)
    
    # Prepare training data
    data = prepare_training_data(intents)
    
    # Preprocess the questions
    data['processed_question'] = data['question'].apply(preprocess_text)
    
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Transform the processed questions into TF-IDF features
    X = vectorizer.fit_transform(data['processed_question'])
    
    # Use the 'intent' column as the target variable
    y = data['intent']
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # Prevent convergence warnings
    model.fit(X, y)
    
    # Save the model and vectorizer
    with open(model_path, 'wb') as f:
        pickle.dump((vectorizer, model), f)
    
    print(f"✅ Model trained and saved successfully at '{model_path}'.")

if __name__ == '__main__':
    train_model()
