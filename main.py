import utils.clients
import utils.http_session
import utils.transport
import utils.session
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from gtts import gTTS
from transformers import pipeline
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
import re
import mutagen
import argparse
from huggingface_hub import snapshot_download
import torch
import numpy as np
# import soundfile as sf
# from fairseq import checkpoint_utils
# import fairseq
# import torchaudio

# === Clean filename from special characters ===
def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

# === 0. Check for ffmpeg ===
def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("❌ ffmpeg not found. Install FFmpeg and add it to PATH.")

# === Get audio duration ===
def get_audio_duration(path):
    from mutagen.mp3 import MP3
    return MP3(path).info.length

# === Adjust audio speed to target duration ===
def adjust_audio_speed(input_path, output_path, target_duration):
    actual_duration = get_audio_duration(input_path)
    if actual_duration > target_duration:
        speed = actual_duration / target_duration
        if speed > 2.0:
            print(f"⚠️ Audio too long: {actual_duration:.2f}s > {target_duration:.2f}s, limited to 2x")
            speed = 2.0
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter:a", f"atempo={speed:.3f}",
            output_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    else:
        shutil.copy(input_path, output_path)
        return False

# === Language Support ===
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'ru': 'Russian',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese'
}

# Language code mapping for different models
LANGUAGE_CODES = {
    'whisper': {
        'en': 'en',
        'ru': 'ru',
        'es': 'es',
        'fr': 'fr',
        'de': 'de',
        'it': 'it',
        'pt': 'pt',
        'ja': 'ja',
        'ko': 'ko',
        'zh': 'zh'
    },
    'm2m100': {
        'en': 'en',
        'ru': 'ru',
        'es': 'es',
        'fr': 'fr',
        'de': 'de',
        'it': 'it',
        'pt': 'pt',
        'ja': 'ja',
        'ko': 'ko',
        'zh': 'zh'
    },
    'gtts': {
        'en': 'en',
        'ru': 'ru',
        'es': 'es',
        'fr': 'fr',
        'de': 'de',
        'it': 'it',
        'pt': 'pt',
        'ja': 'ja',
        'ko': 'ko',
        'zh': 'zh'
    }
}

# === 1. Video transcription with Whisper ===
def transcribe_video(video_path, transcript_path, source_lang='en'):
    print("🔍 Loading Whisper model...")
    model = whisper.load_model("base")
    print("📼 Starting transcription...")
    result = model.transcribe(video_path, language=LANGUAGE_CODES['whisper'][source_lang])
    with open(transcript_path, "w", encoding="utf-8") as f:
        for seg in result['segments']:
            f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")
    print(f"✅ Transcribed {len(result['segments'])} segments")
    return result['segments']

# === 2. Text translation ===
def translate_text(texts, output_path, source_lang='en', target_lang='ru'):
    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    except ImportError:
        raise ImportError("❌ Required packages not found. Install them: pip install transformers sentencepiece")

    print("🌐 Loading translation model...")
    model_name = "facebook/m2m100_418M"
    
    # Set custom cache directory if specified
    cache_dir = os.getenv('HF_CACHE_DIR', os.path.expanduser('~/.cache/huggingface/hub'))
    print(f"📥 Downloading model to: {cache_dir}")
    
    # Download model with progress bar
    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        local_files_only=False,
        resume_download=True
    )
    
    print("🔄 Loading tokenizer and model...")
    tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    
    print("🔁 Translating text...")
    translations = []
    for i, text in enumerate(texts):
        try:
            # Set source and target languages
            tokenizer.src_lang = LANGUAGE_CODES['m2m100'][source_lang]
            tokenizer.tgt_lang = LANGUAGE_CODES['m2m100'][target_lang]
            
            # Tokenize the text
            encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate translation
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(LANGUAGE_CODES['m2m100'][target_lang]),
                max_length=512,
                num_beams=5,
                length_penalty=0.6
            )
            
            # Decode the translation
            translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            print(f"Original ({SUPPORTED_LANGUAGES[source_lang]}): {text[:50]}...")
            print(f"Translated ({SUPPORTED_LANGUAGES[target_lang]}): {translated[:50]}...")
            
        except Exception as e:
            print(f"⚠️ Translation error for segment {i}: {str(e)}")
            print(f"Text: {text[:100]}...")
            translated = text
        translations.append(translated)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in translations:
            f.write(line + "\n")

    print(f"✅ Translated {len(translations)} segments")
    return translations

# === 3. Generate TTS with timing ===
def generate_tts_with_timing(texts, output_dir, base_name, segments):
    audio_files = []

    for i, (translated_text, seg) in enumerate(zip(texts, segments)):
        idx = f"{i:04d}"
        temp_raw = output_dir / f"{base_name}_{idx}_raw.mp3"
        temp_path = output_dir / f"{base_name}_{idx}.mp3"

        # Generate TTS
        generate_tts_audio(translated_text, str(temp_raw))

        # Calculate duration of current video segment
        start, end = float(seg['start']), float(seg['end'])
        target_duration = end - start

        # Adjust audio speed to sync with video
        adjust_audio_speed(str(temp_raw), str(temp_path), target_duration)
        
        audio_files.append(str(temp_path))

    return audio_files

# === RVC Voice Conversion ===
class RVCConverter:
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🎵 Using device: {self.device}")
        
        # Load RVC model
        self.model = self.load_rvc_model(model_path)
        
    def load_rvc_model(self, model_path):
        # Load your RVC model here
        # This is a placeholder - you'll need to implement the actual model loading
        # based on your specific RVC model format
        pass
        
    def convert_voice(self, audio_path, output_path):
        try:
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            audio = audio.to(self.device)
            
            # Process with RVC
            # This is a placeholder - implement actual RVC processing
            converted_audio = self.process_with_rvc(audio)
            
            # Save converted audio
            torchaudio.save(output_path, converted_audio.cpu(), sr)
            return True
        except Exception as e:
            print(f"⚠️ RVC conversion error: {e}")
            return False
            
    def process_with_rvc(self, audio):
        # Placeholder for RVC processing
        # Implement actual RVC processing here
        return audio

# === 4. TTS with gTTS and RVC ===
def generate_tts_audio(text, output_path, lang='ru', use_rvc=True):
    try:
        # Generate initial TTS
        tts = gTTS(text, lang=LANGUAGE_CODES['gtts'][lang])
        temp_path = output_path + ".temp.mp3"
        tts.save(temp_path)
        
        if use_rvc:
            # Convert with RVC
            rvc = RVCConverter("path/to/your/rvc/model")
            success = rvc.convert_voice(temp_path, output_path)
            if not success:
                shutil.copy(temp_path, output_path)
            os.remove(temp_path)
        else:
            shutil.move(temp_path, output_path)
            
        return True
    except Exception as e:
        print(f"⚠️ TTS error for \"{text[:30]}...\": {e}")
        return False

# === 5. Combine mp3 files ===
def combine_audio_segments(audio_dir, final_audio_dir, output_path):
    # Get list of mp3 files
    print("try search in audio_dir=", audio_dir)
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".mp3") and not f.endswith("_raw.mp3")])

    # Create list file for concatenation
    list_path = os.path.join(audio_dir, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for file in audio_files:
            file_path = os.path.join(file)
            f.write(f"file '{file_path}'\n")

    # Convert final file path to absolute path
    print("Converting final file path to absolute path")
    print(final_audio_dir, "/", output_path)
    combined_path = os.path.join(final_audio_dir, output_path)
    os.makedirs(final_audio_dir, exist_ok=True)

    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy", combined_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Combined audio file saved as {combined_path}")

# === 6. Replace audio track in video ===
def replace_audio(video_path, audio_path, output_path):
    print("🎬 Adding audio to video...")

    # ✅ Convert paths to POSIX (with forward slashes)
    video_path = Path(video_path).as_posix()
    audio_path = Path(audio_path).as_posix()
    output_path = Path(output_path).as_posix()

    command = f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}"'
    os.system(command)

# === Main function ===
def main(video_path, source_lang='en', target_lang='ru', use_rvc=True):
    check_ffmpeg()

    input_path = Path(video_path)
    base_name = sanitize_filename(input_path.stem)
    
    # Create output directory structure
    output_dir = Path("output") / base_name
    tts_chunks_dir = output_dir / "tts-chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    tts_chunks_dir.mkdir(parents=True, exist_ok=True)

    # Define all output file paths
    transcript_file = output_dir / "transcript.txt"
    translated_file = output_dir / "translated.txt"
    dubbed_audio = output_dir / "audio_dubbed.mp3"
    final_output = output_dir / f"{base_name}_dubbed.mp4"

    # === Transcription ===
    if transcript_file.exists():
        print(f"⏩ Skipping transcription — found file {transcript_file}")
        segments = []
        with open(transcript_file, "r", encoding="utf-8") as f:
            for line in f:
                if "] " in line:
                    timing, text = line.split("] ", 1)
                    start, end = timing.strip("[]").split(" - ")
                    segments.append({
                        "text": text.strip(),
                        "start": float(start),
                        "end": float(end)
                    })
    else:
        print("🔍 Transcribing video...")
        segments = transcribe_video(video_path, transcript_file, source_lang)

    texts = [seg['text'] for seg in segments]

    # === Translation ===
    if translated_file.exists():
        print(f"⏩ Skipping translation — found file {translated_file}")
        with open(translated_file, "r", encoding="utf-8") as f:
            translations = [line.strip() for line in f.readlines()]
    else:
        print("🌐 Translating text...")
        translations = translate_text(texts, translated_file, source_lang, target_lang)

    # === TTS with timing ===
    expected_audio_files = [str(tts_chunks_dir / f"{base_name}_{i:04d}.mp3") for i in range(len(translations))]
    if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in expected_audio_files):
        print("⏩ Skipping TTS audio generation — all audio files found in folder.")
        temp_audio_files = expected_audio_files
    else:
        print("🔊 Generating TTS audio with timing...")
        temp_audio_files = generate_tts_with_timing(translations, tts_chunks_dir, base_name, segments)
        print(f"✅ Generated {len(temp_audio_files)} audio files")

    # === Combine audio ===
    if not dubbed_audio.exists():
        print("🎧 Combining audio segments...")
        combine_audio_segments(tts_chunks_dir, output_dir, "audio_dubbed.mp3")
    else:
        print(f"⏩ Skipping combination — found file {dubbed_audio}")

    # === Build final video ===
    if not final_output.exists():
        print("🎬 Creating final video...")
        replace_audio(video_path, dubbed_audio, final_output)
    else:
        print(f"⏩ Skipping build — found file {final_output}")

    print(f"🎉 Done! Result: {final_output}")

# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate video between different languages')
    parser.add_argument('video_path', help='Path to the video file to translate')
    parser.add_argument('--source-lang', '-s', choices=SUPPORTED_LANGUAGES.keys(), default='en',
                      help='Source language (default: en)')
    parser.add_argument('--target-lang', '-t', choices=SUPPORTED_LANGUAGES.keys(), default='ru',
                      help='Target language (default: ru)')
    parser.add_argument('--no-rvc', action='store_true', help='Disable RVC voice conversion')
    args = parser.parse_args()
    
    print(f"🌍 Translating from {SUPPORTED_LANGUAGES[args.source_lang]} to {SUPPORTED_LANGUAGES[args.target_lang]}")
    main(args.video_path, args.source_lang, args.target_lang, not args.no_rvc)
