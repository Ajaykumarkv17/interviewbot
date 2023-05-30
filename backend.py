from flask import Flask, request, jsonify,render_template
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import PunktSentenceTokenizer
from flask_cors import CORS
import json
import random

app = Flask(__name__)
CORS(app)
# Load intents from intents.json file
with open('intents.json') as file:
    intents = json.load(file)

# Load chat history for contextual understanding
chat_history = []

# Preprocess user input
def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tag(tokens)]
    
    return tokens

# Map POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Calculate semantic similarity between two sentences using WordNet
def calculate_similarity(sentence1, sentence2):
    # Preprocess the sentences
    tokens1 = preprocess(sentence1)
    tokens2 = preprocess(sentence2)
    
    # Create word sets
    word_set1 = set(tokens1)
    word_set2 = set(tokens2)
    
    # Calculate similarity using Jaccard similarity coefficient
    similarity = len(word_set1.intersection(word_set2)) / len(word_set1.union(word_set2))
    return similarity

# Find the best matching intent based on semantic similarity
def find_best_intent(user_message):
    max_similarity = 0
    best_intent = None
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            similarity = calculate_similarity(user_message, pattern)
            if similarity > max_similarity:
                max_similarity = similarity
                best_intent = intent
    
    return best_intent

# Generate a random response from the matched intent
def generate_response(intent):
    responses = intent['responses']
    return random.choice(responses)

# Perform sentiment analysis on user message
def perform_sentiment_analysis(user_message):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(user_message)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.5:
        sentiment = 'positive'
    elif compound_score <= -0.5:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment

# Extract named entities from user message
def extract_named_entities(user_message):
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(user_message)
    
    named_entities = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        named_entities.extend([token for token, tag in tagged if tag.startswith('NNP')])
    
    return named_entities

# Process user input and generate bot response
def process_user_message(user_message):
    intent = find_best_intent(user_message)
    
    if intent is not None:
        response = generate_response(intent)
    else:
        response = "I'm sorry, I didn't understand that. Can you please rephrase it?"
    
    sentiment = perform_sentiment_analysis(user_message)
    named_entities = extract_named_entities(user_message)
    
    # Store the user message and bot response in the chat history
    chat_history.append({'user_message': user_message, 'bot_response': response})
    
    # You can perform additional actions based on sentiment or named entities if needed
    
    return response

# Handle POST requests to /backend endpoint
@app.route('/backend', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data['message']
    
    bot_response = process_user_message(user_message)
    
    return jsonify({'message': bot_response})

if __name__ == '__main__':

    app.run()