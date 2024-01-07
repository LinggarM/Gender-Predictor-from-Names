from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

def load_vectorizer():
	tfidf_vectorizer = joblib.load('models/tfidf_vectorizer_model.pkl')
	return tfidf_vectorizer

def load_predictor():
	predictor_model = joblib.load('models/predictor_model.pkl')
	return predictor_model

def predict_gender(name):
	# Load vectorizer
	tfidf_vectorizer = load_vectorizer()

	# Load predictor
	predictor_model = load_predictor()

	# Vectorize name
	name_vectorized = tfidf_vectorizer.transform(name)
	
	# Predict Gender
	pred = predictor_model.predict(name_vectorized)

	# Return prediction
	pred_gender = "Male" # Default value
	if pred == 0:
		pred_gender = "Female"
	else:
		pred_gender = "Male"
	return pred_gender

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict_gender', methods=['POST'])
def predict_gender_route():
    name = request.form.get('name')
    prediction = predict_gender([name])
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
 	app.run(debug= True)