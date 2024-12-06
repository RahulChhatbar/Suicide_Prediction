from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')  # Serve the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    probability = data.get('probability', 0)

    # Process the input as needed (this is where you would use your logic)
    if probability > 50:
        predicted_class = 'positive'
    elif probability < 50:
        predicted_class = 'negative'
    else:
        predicted_class = 'invalid'

    response = {
        'predicted_class': predicted_class,
        'formatted_probability': round(probability, 2)
    }
    return jsonify(response)

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('.', path)  # Serve static files (CSS, JS)

if __name__ == '__main__':
    app.run(debug=True)
