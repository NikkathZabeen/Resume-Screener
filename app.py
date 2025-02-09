from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import nltk

# Ensure required NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt_tab')
# Load your saved model and TF-IDF vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Define an education mapping that matches your training
education_levels = {
    "PhD": 0,
    "MSc": 1,
    "BSc": 2
}

# Define text preprocessing function
def preprocess_text(text):
    if text is None:
        return ""
    text = text.lower()                                 # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)                 # Remove punctuation
    text = re.sub(r'\d+', '', text)                     # Remove numbers
    tokens = word_tokenize(text)                        # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Initialize the Flask app
app = Flask(__name__)

# Home route to render the input form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to process form data and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    skills = request.form['skills']
    experience = float(request.form['experience'])
    education = request.form['education']
    
    # Preprocess the skills text and convert to numerical features
    cleaned_skills = preprocess_text(skills)
    skills_features = vectorizer.transform([cleaned_skills]).toarray()
    
    # Map education level to numerical value (default to 0 if not found)
    education_num = education_levels.get(education, 0)
    
    # Combine features: TF-IDF features + numeric features for experience and education
    input_features = np.hstack((skills_features, [[experience, education_num]]))
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_features)[0]
    result = "Hired" if prediction == 1 else "Not Hired"
    
    # Render the result back on the form page
    return render_template('index.html', prediction_text=f"The applicant is predicted: {result}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
