import openai
from flask import Flask, render_template, request, jsonify

app = Flask(_name)  # Flask initialization without __name_

# Replace with your actual OpenAI API key
openai.api_key = "your_openai_api_key_here"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def chatbot_response():
    user_message = request.args.get('msg')

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are ASA, a smart assistant trained to solve real-world problems."},
            {"role": "user", "content": user_message}
        ]
    )

    return jsonify(response=response['choices'][0]['message']['content'])

# Without using '_name_', simply run the app directly
if _name_ == '_main_':
    app.run(debug=True)