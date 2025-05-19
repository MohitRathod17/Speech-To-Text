import streamlit as st
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io

# Load model & processor (cache them to avoid reloading every time)
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = load_model()

# Function to transcribe audio
def transcribe_audio(audio_file):
    # Reset pointer and read raw bytes
    audio_file.seek(0)
    audio_bytes = io.BytesIO(audio_file.read())

    # Use soundfile to read waveform and sample rate
    waveform_np, sample_rate = sf.read(audio_bytes)

    # Convert to mono if stereo
    if len(waveform_np.shape) > 1:
        waveform_np = waveform_np.mean(axis=1)

    # Convert NumPy array to float32 PyTorch tensor
    waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)

    # Resample if needed (Wav2Vec2 expects 16kHz)
    if sample_rate != 16000:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Tokenize and get model predictions
    inputs = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

# Streamlit UI
st.title("🎙️ Speech-to-Text with Wav2Vec2")
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.info("Transcribing...")
    try:
        text = transcribe_audio(uploaded_file)
        st.success("Transcription:")
        st.write(text)
    except Exception as e:
        st.error(f"Error: {e}")
