from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import os
from src.train import train_model
from src.generate import generate_text
from src.pre_processor import PoemPreprocessor, load_corpus

app = Flask(__name__)
socketio = SocketIO(app)

# Global variable to keep track of the training thread
training_thread = None
training_active = False

@app.route('/')
def index():
    global training_active
    training_active = False  # Reset training status on page load
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/generate', methods=['POST'])
def generate():
    global training_thread, training_active
    start_text = request.form['start_text']
    poet = request.form['poet']
    
    # Load the corpus based on selected poet
    if poet == "Hemingway":
        corpus_path = "data/hemingway.txt"
    elif poet == "Frost":
        corpus_path = "data/frost.txt"
    else:
        return jsonify({'poem': "sorry, that poet's style is not available right now."})

    # Cancel any ongoing training
    if training_thread and training_thread.is_alive():
        training_active = False
        training_thread.join()

    # Start a new training thread
    training_active = True
    training_thread = threading.Thread(target=run_training, args=(corpus_path, start_text))
    training_thread.start()

    return jsonify({'status': 'Training started'})

def run_training(corpus_path, start_text):
    global training_active
    corpus = load_corpus(corpus_path)
    preprocessor = PoemPreprocessor(corpus, user_input=start_text)
    
    def progress_callback(epoch, total_epochs, step, total_steps, loss):
        global training_active
        if not training_active:
            raise KeyboardInterrupt  # Stop training if the flag is set to False
        progress = (epoch * total_steps + step) / (total_epochs * total_steps)
        socketio.emit('training_progress', {'progress': progress, 'epoch': epoch + 1, 'loss': loss})

    try:
        model, preprocessor = train_model(preprocessor=preprocessor, progress_callback=progress_callback)
        generated_text = generate_text(model, start_text, preprocessor, num_words=120, sequence_length=10)
        socketio.emit('training_complete', {'poem': generated_text})
    except KeyboardInterrupt:
        socketio.emit('training_cancelled', {'message': 'training was cancelled due to page refresh.'})
    
    training_active = False  # Reset the training status after completion

if __name__ == '__main__':
    socketio.run(app, debug=True)
