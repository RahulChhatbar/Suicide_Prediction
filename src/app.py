from flask import Flask, request, jsonify, send_from_directory
from test import test_statement

app = Flask(__name__)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'input_text' not in data:
        return jsonify({'error': 'No input_text provided'}), 400
    
    input_text = data['input_text']
    
    try:
        response = test_statement(input_text)
        suicide_percentage = response.get('suicide_percentage')
        predicted_class = response.get('ensemble_prediction')
        
        if suicide_percentage is None:
            return jsonify({'error': 'No suicide percentage found in the response'}), 500

        return jsonify({
            'suicide_percentage': suicide_percentage,
            'predicted_class': predicted_class
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True)
