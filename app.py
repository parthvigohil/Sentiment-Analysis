from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import pickle
import os
from twitter_scraper_selenium import scrape_profile
import json
#import pandas as pd
#from ntscraper import Nitter
#from twscrape import API, gather
#from twscrape.logger import set_log_level
import requests
import joblib

app = Flask(__name__, template_folder='templates')

model = tf.keras.models.load_model('cnn_bidirectional_lstm_model6')
tokenizer_path = os.path.join(os.getcwd(), 'tokenizer.pickle')
# Load the tokenizer configuration from the file
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)
max_sequence_length = 100 
#scraper = Nitter(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/searchtweets', methods=["POST", "GET"])
def search():
        if request.method == 'POST':
            # If the form is submitted, get the input text from the form
            input_text = request.form['input_text']
            # Use the predict_sentiment function to make predictions
            prediction = predict_sentiment(input_text, model, tokenizer, max_sequence_length)
            # Return the prediction to be displayed on the page
            return render_template('searchtwt.html', prediction=prediction, input_text=input_text)
        else:
            # If it's a GET request, just render the home.html template
            return render_template('searchtwt.html', prediction=None, input_text=None)

def predict_sentiment(text, model, tokenizer, max_sequence_length):
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad sequences to the same length as during training
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    # Predict sentiment
    prediction = model.predict(padded_sequences)
    # Convert prediction to sentiment label
    print(prediction)
    if prediction <= 0.40:
        return 'Negative'
    if prediction >= 0.60:
        return 'Positive'
    else:
        return 'Neutral'

@app.route('/searchusers', methods=['POST', 'GET'])
def search_tweets():
     
     if request.method == 'POST':

        username = request.form['username']
        username =username.lower()
        api_key = '660194bbf8703856fa5c0b7a'
        url = 'https://twitter.com/' +username
        parsed = 'true'

        params = {
        'api_key': api_key,
        'url': url,
        'parsed': parsed
        }
        response = requests.get('https://api.scrapingdog.com/twitter', params=params)

        if response.status_code == 200:
            response_data = response.json()

        results = []
        for data in response_data:
            if 'tweet' in data:
                results.append(data['tweet'])

        display = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        for tweet in results:
            sentiment = predict_sentiment(tweet, model, tokenizer, max_sequence_length)
            display.append({'text': tweet, 'sentiment': sentiment})
            # Update counts
            if sentiment == 'Positive':
                positive_count += 1
            elif sentiment == 'Negative':
                negative_count += 1
            else:
                neutral_count += 1
                
        # Display sentiment analysis results on the web page
        return render_template('searchuser.html', display=display, positive_count=positive_count, negative_count=negative_count, neutral_count=neutral_count)
     
     else:
         return render_template('searchuser.html', display=None, input_text=None)
     
@app.route('/searchkeywords', methods=['POST', 'GET'])
def search_keywords():
     
     if request.method == 'POST':

        keyword = request.form['keyword']
        keyword = keyword.lower()
        #output = scrape_profile(twitter_usernameusername, browser="chrome", tweets_count=5, headless=False)
        api_key = '660194bbf8703856fa5c0b7a'
        url = 'https://twitter.com/' +keyword
        parsed = 'true'
        
        params = {
        'api_key': api_key,
        'url': url,
        'parsed': parsed
        }
        response = requests.get('https://api.scrapingdog.com/twitter', params=params)

        if response.status_code == 200:
            response_data = response.json()

        #tweet_data = json.loads(response_data)
        results = []
        #for tweet_id, tweet_info in tweet_data.items():
        #    results.append(tweet_info['content'])

        for data in response_data:
            if 'tweet' in data:
                results.append(data['tweet'])

        #tweets = fetch_tweetsusername)[:10]
        #tweets = scraper.get_tweetsusername, number = 10)
       #print(tweets)

        display = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        for tweet in results:
            #text = tweet.get('text', '')
            sentiment = predict_sentiment(tweet, model, tokenizer, max_sequence_length)
            display.append({'text': tweet, 'sentiment': sentiment})
            # Update counts
            if sentiment == 'Positive':
                positive_count += 1
            elif sentiment == 'Negative':
                negative_count += 1
            else:
                neutral_count += 1
                
        # Display sentiment analysis results on the web page
        return render_template('searchkey.html', display=display, positive_count=positive_count, negative_count=negative_count, neutral_count=neutral_count)
        #return response
     
     else:
         return render_template('searchkey.html', display=None, input_text=None)
     
if __name__ == '__main__':
    app.run(debug=True)