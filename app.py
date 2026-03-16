from flask import Flask, render_template, request, jsonify
import whisper
from transformers import pipeline
import os

app = Flask(__name__)

print("Loading models...")

whisper_model = whisper.load_model("base")
sentiment_model = pipeline("sentiment-analysis")

print("Models loaded successfully")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyse", methods=["POST"])
def analyse():

    if "audio" not in request.files:
        return jsonify({"error":"No audio uploaded"})

    audio_file = request.files["audio"]
    path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(path)

    # Speech to text
    result = whisper_model.transcribe(path)
    text = result["text"]

    # Sentiment
    sentiment = sentiment_model(text)[0]

    return jsonify({
        "transcript": text,
        "sentiment": sentiment["label"],
        "score": round(sentiment["score"]*100,2)
    })

if __name__ == "__main__":
    app.run(debug=True)