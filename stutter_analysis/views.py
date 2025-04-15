from django.shortcuts import render

import librosa
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import whisper
from scipy.signal import butter, lfilter
import noisereduce as nr
from g2p_en import G2p
from collections import Counter
import re
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_audio(y, sr, target_sr=16000):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    y_denoised = butter_lowpass_filter(y_denoised, cutoff=4000, fs=sr)
    return y_denoised, sr

def extract_features_from_segment(y, sr, num_mfcc=40):
    if len(y) < sr * 0.1:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    try:
        pitch = np.mean(librosa.yin(y, fmin=50, fmax=500))
    except:
        pitch = 0.0
    features = np.concatenate(([zcr, spectral_centroid, pitch], mfcc_mean))
    return features

def segment_audio_by_words(y, sr, word_timestamps):
    segments = []
    for word_info in word_timestamps:
        start_time = word_info["start"]
        end_time = word_info["end"]
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]
        segments.append((segment, start_time, end_time, word_info["word"]))
    return segments

def detect_stuttering_per_segment(segments, sr, scaler, model, threshold=0.98):
    stuttered_segments = []
    for segment, start_time, end_time, word in segments:
        features = extract_features_from_segment(segment, sr)
        if features is None:
            continue
        if features.shape[0] != 43:
            continue
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled, verbose=0)[0][0]
        if prediction >= threshold:
            stuttered_segments.append((start_time, end_time, word, prediction))
    return stuttered_segments

def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    word_timestamps = []
    for segment in result["segments"]:
        word_timestamps.extend(segment.get("words", []))
    return result["text"], word_timestamps

def highlight_stuttered_words(full_text, stuttered_segments, word_timestamps):
    highlighted_words = []
    for word_info in word_timestamps:
        word = word_info["word"]
        word_start = word_info["start"]
        word_end = word_info["end"]
        is_stuttered = any(
            s_word.lower() == word.lower() and 
            max(s_start, word_start) <= min(s_end, word_end)
            for s_start, s_end, s_word, _ in stuttered_segments
        )
        highlighted_words.append(f"<b>{word}</b>" if is_stuttered else word)
    return " ".join(highlighted_words)

def get_phonemes(word, g2p):
    clean_word = re.sub(r'[.,!?]', '', word).strip().lower()
    if not clean_word:
        return []
    if clean_word == "20th":
        clean_word = "twentieth"
    elif clean_word == "20":
        clean_word = "twenty"
    elif clean_word == "a":
        clean_word = "a"
    try:
        phonemes = g2p(clean_word)
        phonemes = [re.sub(r'[0-2]', '', p) for p in phonemes]
        phonemes = [p for p in phonemes if p and p.isalpha()]
        consonants = {'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'}
        phonemes = [p for p in phonemes if p in consonants]
        return phonemes
    except:
        return []

def find_most_stuttered_phoneme(stuttered_segments):
    if not stuttered_segments:
        return None, None, 0, {}
    g2p = G2p()
    all_phonemes = []
    word_counts = Counter([word for _, _, word, _ in stuttered_segments])
    for word, count in word_counts.items():
        phonemes = get_phonemes(word, g2p)
        all_phonemes.extend(phoneme for phoneme in phonemes for _ in range(count))
    if not all_phonemes:
        return None, None, 0, {}
    phoneme_counts = Counter(all_phonemes)
    most_common_phoneme, count = phoneme_counts.most_common(1)[0]
    return most_common_phoneme, None, count, phoneme_counts

def plot_stutter_frequency(stuttered_segments, duration, bin_size=1.0):
    if not stuttered_segments:
        return None
    start_times = [start for start, _, _, _ in stuttered_segments]
    max_time = max(duration, max(start_times, default=0))
    if max_time <= 0:
        return None
    bins = np.arange(0, max_time + bin_size, bin_size)
    stutter_counts, _ = np.histogram(start_times, bins=bins)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bins[:-1], stutter_counts, width=bin_size, align='edge')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of Stutters')
    ax.set_title('Stutter Frequency Over Time')
    ax.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

def calculate_stutter_rate(stuttered_segments, word_timestamps, duration):
    num_stutters = len(stuttered_segments)
    num_words = len(word_timestamps)
    word_rate = (num_stutters / num_words * 100) if num_words > 0 else 0
    time_rate = (num_stutters / (duration / 60)) if duration > 0 else 0
    return word_rate, time_rate

def generate_practice_passage(phoneme):
    try:
        phoneme_sound = phoneme.lower()
        if phoneme == 'CH':
            phoneme_sound = 'ch'
        elif phoneme == 'DH':
            phoneme_sound = 'th (as in this)'
        elif phoneme == 'JH':
            phoneme_sound = 'j'
        elif phoneme == 'NG':
            phoneme_sound = 'ng'
        elif phoneme == 'SH':
            phoneme_sound = 'sh'
        elif phoneme == 'TH':
            phoneme_sound = 'th (as in think)'
        elif phoneme == 'ZH':
            phoneme_sound = 'zh (as in vision)'
        prompt = (
            f"Create a short, creative story (3-5 sentences) for speech therapy practice, "
            f"emphasizing words starting with or containing the consonant sound /{phoneme_sound}/. "
            f"Use at least 5 distinct words with /{phoneme_sound}/ (e.g., for /n/: night, name, nest), "
            f"highlight these words with asterisks (*night*), and keep the tone engaging and suitable for practice."
        )
        client = ollama.Client(host='http://127.0.0.1:11434')
        response = client.generate(model='phi3', prompt=prompt)
        passage = response['response'].strip()
        if not '*' in passage:
            sample_words = {
                'B': ['big', 'ball', 'bird'], 'CH': ['chair', 'cheese', 'child'], 
                'D': ['dog', 'day', 'dance'], 'DH': ['this', 'that', 'there'],
                'F': ['fish', 'friend', 'fire'], 'G': ['game', 'girl', 'gift'],
                'HH': ['hat', 'house', 'hill'], 'JH': ['jump', 'joy', 'juice'],
                'K': ['key', 'cake', 'car'], 'L': ['love', 'light', 'leaf'],
                'M': ['moon', 'man', 'music'], 'N': ['night', 'name', 'nest'],
                'NG': ['song', 'ring', 'wing'], 'P': ['pen', 'park', 'play'],
                'R': ['red', 'road', 'river'], 'S': ['sun', 'star', 'smile'],
                'SH': ['ship', 'shoe', 'shop'], 'T': ['time', 'tree', 'table'],
                'TH': ['think', 'thing', 'three'], 'V': ['voice', 'view', 'vine'],
                'W': ['water', 'wind', 'way'], 'Y': ['year', 'yard', 'yellow'],
                'Z': ['zoo', 'zebra', 'zone'], 'ZH': ['vision', 'measure', 'treasure']
            }
            words = sample_words.get(phoneme, ['word', 'sound'])
            passage = f"A *{words[0]}* found a *{words[1]}* to practice *{phoneme_sound}* sounds."
        return passage
    except Exception as e:
        return f"Could not generate passage: {str(e)}. Try practicing with words like *sound* and *speech*."

def index(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'Uploads'))
        filename = fs.save(audio_file.name, audio_file)
        audio_path = os.path.join(settings.MEDIA_ROOT, 'Uploads', filename)
        
        try:
            # Load model and scaler
            scaler = joblib.load(os.path.join(settings.BASE_DIR, 'standard_scaler.save'))
            model = load_model(os.path.join(settings.BASE_DIR, 'Stutter_Detection_Model.h5'))
            
            # Process audio
            y, sr = librosa.load(audio_path, sr=None)
            y, sr = preprocess_audio(y, sr, target_sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Transcribe audio
            full_text, word_timestamps = transcribe_audio_whisper(audio_path)
            
            # Analyze stuttering
            segments = segment_audio_by_words(y, sr, word_timestamps)
            stuttered_segments = detect_stuttering_per_segment(segments, sr, scaler, model, threshold=0.98)
            
            # Highlight stuttered words
            highlighted_transcription = highlight_stuttered_words(full_text, stuttered_segments, word_timestamps)
            
            # Stutter frequency plot
            plot_data = plot_stutter_frequency(stuttered_segments, duration, bin_size=1.0)
            
            # Stutter rate
            word_rate, time_rate = calculate_stutter_rate(stuttered_segments, word_timestamps, duration)
            
            # Phoneme analysis
            phoneme, _, _, _ = find_most_stuttered_phoneme(stuttered_segments)
            practice_passage = generate_practice_passage(phoneme) if phoneme else "No consonant phonemes identified in stuttered words."
            
            context.update({
                'full_text': full_text,
                'highlighted_transcription': highlighted_transcription,
                'plot_data': plot_data,
                'word_rate': f"{word_rate:.2f}",
                'time_rate': f"{time_rate:.2f}",
                'phoneme': phoneme,
                'practice_passage': practice_passage,
                'audio_url': fs.url(filename),
            })
            
        except Exception as e:
            context['error'] = str(e)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
    return render(request, 'stutter_analysis/index.html', context)