from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load models and scaler
with open('notebook/linear_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('notebook/lasso_model.pkl', 'rb') as f:
    lasso_model = pickle.load(f)

with open('notebook/minmax.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Home route
@app.route('/')
def home():
    return '''
        <h1>Crypto Price Prediction</h1>
        <form method="POST" action="/predict">
            <p>Enter comma-separated feature values (same order as training):</p>
            <input type="text" name="features" style="width:400px">
            <input type="submit" value="Predict">
        </form>
    '''

# Prediction route (web form)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and preprocess features
        features = [float(x) for x in request.form['features'].split(',')]
        features_array = np.array([features])
        scaled_features = scaler.transform(features_array)

        # Get predictions
        linear_pred = linear_model.predict(scaled_features)[0]
        lasso_pred = lasso_model.predict(scaled_features)[0]

        return f'''
            <h2>Prediction Results</h2>
            <p><strong>Linear Regression:</strong> {linear_pred:.4f}</p>
            <p><strong>Lasso Regression:</strong> {lasso_pred:.4f}</p>
            <a href="/">Go Back</a>
        '''
    except Exception as e:
        return f"Error: {str(e)}"

# API endpoint for JSON input
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    try:
        features = np.array([data['features']])
        scaled_features = scaler.transform(features)
        linear_pred = linear_model.predict(scaled_features)[0]
        lasso_pred = lasso_model.predict(scaled_features)[0]

        return jsonify({
            'linear_prediction': linear_pred,
            'lasso_prediction': lasso_pred
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
