# Import necessary libraries from Flask and our custom engine
from flask import Flask, render_template, request, jsonify 
from engine import extract_text_from_pdf, analyze_resume
import traceback # This is a tool to help us see detailed errors

# Initialize the Flask application
app = Flask(__name__)

# --- Route 1: The Home Page ---
# This function serves our main professional landing page
@app.route('/')
def home():
    """
    Renders our main professional landing page, home.html.
    """
    return render_template('home.html')

# --- Route 2: The Analysis API ---
# This is the backend endpoint that our website's JavaScript will call.
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    This version has a "bulletproof" error catcher. It will find
    any hidden problem and report it back to us cleanly in a JSON format.
    """
    try:
        # --- Step 1: Validate the incoming data ---
        if 'resume_pdf' not in request.files:
            return jsonify({'error': 'No resume file was submitted. Please try again.'}), 400
        
        resume_file = request.files['resume_pdf']
        job_description = request.form['job_description']
        company_name = request.form['company_name']

        if resume_file.filename == '':
            return jsonify({'error': 'No resume file was selected. Please try again.'}), 400

        # --- Step 2: Call our powerful engine from engine.py ---
        resume_text = extract_text_from_pdf(resume_file)
        
        if not resume_text or not resume_text.strip():
            return jsonify({'error': 'Could not extract any text from the uploaded PDF. It might be an image-based or corrupted file.'}), 400

        match_score, missing_keywords = analyze_resume(resume_text, job_description)
        
        # --- Step 3: Send a successful response back to the JavaScript ---
        # The data is packaged in a clean JSON format.
        return jsonify({
            'score': match_score,
            'missing': missing_keywords,
            'resume_text': resume_text, # We send this back for the Gemini features
            'job_description': job_description,
            'company_name': company_name
        })

    except Exception as e:
        # --- JANNI'S MASTER DEBUGGER ---
        error_details = traceback.format_exc()
        print("--- A CRITICAL BACKEND ERROR WAS CAUGHT ---")
        print(error_details)
        print("-----------------------------------------")
        return jsonify({
            'error': 'A critical error happened on the server. Janni is investigating.',
            'details': str(e)
            }), 500

# This is the standard block to make the server runnable
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)











