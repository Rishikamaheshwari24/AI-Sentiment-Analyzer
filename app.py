from transformers import pipeline
import gradio as gr
import speech_recognition as sr

# -------------------------
# Load your AI model
# -------------------------
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# -------------------------
# Text sentiment function
# -------------------------
def analyze_text(text):
    if not text:
        return "Please enter some text."
    result = classifier(text)[0]
    return f"Label: {result['label']}, Score: {result['score']:.2f}"

# -------------------------
# Audio sentiment function
# -------------------------
def analyze_audio(audio_file):
    if not audio_file:
        return "Please upload an audio file."
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Speech Recognition service error."
    
    result = classifier(text)[0]
    return f"Transcribed Text: {text}\nLabel: {result['label']}, Score: {result['score']:.2f}"

# -------------------------
# Build Gradio interface
# -------------------------
text_interface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=5, placeholder="Enter your text here..."),
    outputs="text",
    title="AI Sentiment Analyzer (Text)",
    description="Enter text to detect sentiment."
)

audio_interface = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs="text",
    title="AI Sentiment Analyzer (Audio)",
    description="Upload an audio file to detect sentiment."
)

# Combine both tabs
app = gr.TabbedInterface([text_interface, audio_interface], ["Text", "Audio"])

# -------------------------
# Launch app
# -------------------------
app.launch()