import pickle
from chatbot.preprocess import preprocess_text

# Load the model and vectorizer
with open('chatbot/model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

def get_response(user_input):
    processed_input = preprocess_text(user_input)
    X = vectorizer.transform([processed_input])
    response = model.predict(X)
    return response[0]
