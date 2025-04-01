from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import inference
import os
from datetime import datetime
import json

MESSAGES_FILE = 'messages_history.json'

# Khởi tạo server
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET', 'POST'])
def run_server():
    return "Growing with Linh <3"


# Utility function to read messages from the file
def read_messages():
    if not os.path.exists(MESSAGES_FILE):
        return []
    with open(MESSAGES_FILE, 'r', encoding="utf-8") as file:
        return json.load(file)

# Utility function to write messages to the file
def write_messages(messages):
    with open(MESSAGES_FILE, 'w', encoding='utf-8') as file:
        json.dump(messages, file, ensure_ascii=False, indent=2)

# Utility function to clear messages file
def clear_messages():
    if os.path.exists(MESSAGES_FILE):
        os.remove(MESSAGES_FILE)

@app.route('/save_messages_history', methods=['POST', 'GET'])
def save_messages_history():

    global history
    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    messages = read_messages()
    messages.append({
        'message': message,
        'timestamp': datetime.now().isoformat()
    })

    write_messages(messages)
    history = read_messages()

    return history

@app.route('/clear_messages_history', methods=['POST'])
def clear_message_history():
    clear_messages()
    return jsonify({'success': True, 'message': 'Message history cleared'})

@app.route('/inference', methods=['GET', 'POST'])
def inference_():
    return inference(history[-1]['message'])


if __name__ == "__main__":
    app.run(host="0.0.0.0")