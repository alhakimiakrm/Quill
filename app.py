from flask import Flask, render_template, request, jsonify
from src.train import train_model
from src.generate import generate_text
from src.pre_processor import PoemPreprocessor, load_corpus
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    start_text = request.form['start_text']
    poet = request.form['poet']
    
    # Load the corpus based on selected poet (only Hemingway available now)
    if poet == "Hemingway":
        corpus_path = "data/hemingway.txt"
    else:
        return jsonify({'poem': "Sorry, only Hemingway's style is available right now."})

    # Train model based on user input
    corpus = load_corpus(corpus_path)
    preprocessor = PoemPreprocessor(corpus, user_input=start_text)
    model, preprocessor = train_model(preprocessor=preprocessor)

    # Generate poem based on user input
    generated_text = generate_text(model, start_text, preprocessor, num_words=120, sequence_length=10)
    
    return jsonify({'poem': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
