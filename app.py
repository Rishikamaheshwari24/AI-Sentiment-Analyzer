import gradio as gr
import whisper
from transformers import pipeline

# Load models
whisper_model = whisper.load_model("base")
sentiment_model = pipeline("sentiment-analysis")

def analyse_audio(audio_file):
    # audio_file is path
    result = whisper_model.transcribe(audio_file)
    text = result["text"]
    sentiment = sentiment_model(text)[0]
    return text, sentiment["label"], round(sentiment["score"]*100, 2)

# Gradio interface
gr.Interface(
    fn=analyse_audio,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=["text", "text", "number"],
    title="AI Speech Sentiment Analyzer",
    description="Upload an audio file to analyze sentiment"
).launch()