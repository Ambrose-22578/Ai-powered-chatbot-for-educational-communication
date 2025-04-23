from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import difflib
import os
import sqlite3
from datetime import datetime
import threading
import time
import smtplib
import bcrypt  # For password hashing
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
from chatbot.preprocess import preprocess_text
import json
import random

app = Flask(__name__)
app.secret_key = "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"  # Replace with a secure secret key

# Database setup for users
def init_user_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')
    conn.commit()
    conn.close()

init_user_db()  # Ensure database is initialized on startup

# Register route
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data['email']
    password = data['password']

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_password))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Registration successful!'}), 200
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Email already registered!'}), 400

# Login route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data['email']
    password = data['password']

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    conn.close()

    if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
        session['user'] = email  # Store user in session
        return jsonify({'message': 'Login successful!'}), 200
    return jsonify({'message': 'Invalid email or password!'}), 401

# Logout route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login_page'))

# Login & Registration page
@app.route('/login_page')
def login_page():
    return render_template('login_register.html')

# Ensure authentication before accessing chatbot
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for('login_page'))
    return render_template("index.html")

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Define file paths
INTENTS_FILE = "chatbot/data/intents.json"
MODEL_FILE = "chatbot/data/model.pkl"

# Load intents
try:
    with open(INTENTS_FILE, 'r', encoding="utf-8") as f:
        intents = json.load(f)
    print("✅ Intents loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: '{INTENTS_FILE}' not found. Please provide a valid intents file.")
    exit(1)

# Load semantic similarity model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")  # For semantic similarity
qa_pipeline = pipeline("question-answering")  # For fallback responses

# Load pre-trained TF-IDF vectorizer and model
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as model_file:
        vectorizer, model = pickle.load(model_file)
    print("✅ Model and vectorizer loaded successfully!")
else:
    print(f"❌ Error: Model file '{MODEL_FILE}' not found. Train the model using 'python -m chatbot.train'")
    exit(1)

    import re
    
    def make_links_clickable(text):
        """Replace raw URLs in text with clickable HTML links."""
        url_pattern = re.compile(r'https?://\S+')
        return url_pattern.sub(lambda x: f'<a href="{x.group()}" target="_blank">Click here</a>', text)


def get_response(intent):
    """Fetch a random response for the given intent."""
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            return random.choice(intent_data['responses'])
        # Make links clickable
            response = make_links_clickable(response)
            return response
    return "I'm sorry, I don't understand."

def find_best_match(question):
    """Find the closest matching intent using semantic similarity."""
    question_embedding = semantic_model.encode(question)
    best_match = None
    highest_similarity = 0.0

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_embedding = semantic_model.encode(pattern)
            similarity = np.dot(question_embedding, pattern_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(pattern_embedding))
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = intent['tag']

    return best_match if highest_similarity > 0.7 else "fallback"

def predict_intent(question):
    """Predict the best intent using TF-IDF and semantic similarity."""
    processed_question = question.lower()
    X = vectorizer.transform([processed_question])  # Ensure vectorizer is fitted before transforming

    # Predict intent using logistic regression
    predicted_intent = model.predict(X)[0]

    # Compute similarity scores
    intent_vectors = vectorizer.transform([intent['patterns'][0] for intent in intents['intents']])
    similarity_scores = cosine_similarity(X, intent_vectors)

    if max(similarity_scores[0]) < 0.5:
        # If TF-IDF confidence is low, use semantic similarity
        predicted_intent = find_best_match(question)

    return predicted_intent

def chatbot_response(question):
    """Generate a response for the user's question."""
    intent = predict_intent(question)
    if intent == "fallback":
        return "I'm not sure how to answer that. Can you rephrase?"
    return get_response(intent)

# Chatbot route
@app.route('/get_response', methods=['POST'])
def get_bot_response():
    try:
        data = request.get_json()
        user_input = data.get('user_input', '').strip()
        if not user_input:
            return jsonify({'bot_response': 'Please enter a valid question.'})

        # Check for reminder command
        if 'remind me' in user_input.lower():
            bot_response = set_reminder(user_input)
        else:
            # Predict the intent
            intent = predict_intent(user_input)
            # Generate the response
            bot_response = get_response(intent)
        
        # Store the message in chat history
        store_message(user_input, bot_response)
        
        return jsonify({'bot_response': bot_response})
    except Exception as e:
        print("Error:", e)
        return jsonify({'bot_response': 'An error occurred. Please try again later.'})

# Create chat history database
def create_chat_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_message TEXT,
                        bot_response TEXT
                    )''')
    conn.commit()
    conn.close()

create_chat_db()  # Ensure chat history database is created

def store_message(user_message, bot_response):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_message, bot_response) VALUES (?, ?)", (user_message, bot_response))
    conn.commit()
    conn.close()

def get_chat_history():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_message, bot_response FROM chat_history ORDER BY id DESC LIMIT 10")
    history = cursor.fetchall()
    conn.close()
    return history

# Chatbot page (protected)
@app.route('/chatbot')
def chatbot():
    if "user" not in session:
        return redirect(url_for('login_page'))
    return render_template('chatbot.html')

@app.route('/get_history', methods=['GET'])
def get_history():
    history = get_chat_history()
    return jsonify(history)

# Email Reminder System
reminder_file = 'chatbot/data/reminders.csv'
if not os.path.exists(reminder_file):
    pd.DataFrame(columns=['message', 'date', 'time', 'email']).to_csv(reminder_file, index=False)

EMAIL_ADDRESS = "koech011@gmail.com"
EMAIL_PASSWORD = "racs ilhl cogsetqz"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        server.quit()
        print(f"📧 Email sent to {to_email}: {subject}")
    except Exception as e:
        print("Error sending email:", e)

def set_reminder(user_input):
    try:
        parts = user_input.lower().replace('remind me', '').strip().split(' on ')
        message = parts[0].strip()
        datetime_part = parts[1].strip().split(' at ')
        date_str, time_str = datetime_part[0], datetime_part[1]
        email = session.get('user', 'ambrosekoech08@gmail.com')
        
        df = pd.read_csv(reminder_file)
        new_reminder = pd.DataFrame({'message': [message], 'date': [date_str], 'time': [time_str], 'email': [email]})
        df = pd.concat([df, new_reminder], ignore_index=True)
        df.to_csv(reminder_file, index=False)
        
        return f"Reminder set: '{message}' on {date_str} at {time_str}. You will receive an email reminder."
    except:
        return "Invalid reminder format. Please use: 'Remind me [message] on [date] at [time]'."

def check_reminders():
    while True:
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M')
            df = pd.read_csv(reminder_file)
            reminders_to_send = df[(df['date'] + ' ' + df['time']) == now]
            for _, row in reminders_to_send.iterrows():
                send_email(row['email'], "Reminder Alert", f"Reminder: {row['message']}")
            df = df[~((df['date'] + ' ' + df['time']) == now)]
            df.to_csv(reminder_file, index=False)
        except Exception as e:
            print("Error checking reminders:", e)
        time.sleep(30)

reminder_thread = threading.Thread(target=check_reminders, daemon=True)
reminder_thread.start()

@app.route('/about')
def about():
    if "user" not in session:  # Ensure the user is logged in
        return redirect(url_for('login_page'))
    return render_template('about.html')

@app.route('/contact')
def contact():
    if "user" not in session:  # Ensure the user is logged in
        return redirect(url_for('login_page'))
    return render_template('contact.html')

@app.route('/blog')
def blog():
    if "user" not in session:  # Ensure the user is logged in
        return redirect(url_for('login_page'))
    return render_template('blog.html')

@app.route('/course')
def course():
    if "user" not in session:  # Ensure the user is logged in
        return redirect(url_for('login_page'))
    return render_template('course.html')

if __name__ == '__main__':
    app.run(debug=True)