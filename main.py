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
import sys
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
            "-af", "silencedetect=n=-50dB:d=0.1",  # Detect silence
            output_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    elif actual_duration < target_duration * 0.8:  # If audio is too short
        # Add silence at the end to match duration
        silence_duration = target_duration - actual_duration
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"apad=pad_dur={silence_duration}",
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

# === Voice Selection ===
VOICE_OPTIONS = {
    'gtts': {
        'en': {'male': 'en', 'female': 'en'},
        'ru': {'male': 'ru', 'female': 'ru'},
        'es': {'male': 'es', 'female': 'es'},
        'fr': {'male': 'fr', 'female': 'fr'},
        'de': {'male': 'de', 'female': 'de'},
        'it': {'male': 'it', 'female': 'it'},
        'pt': {'male': 'pt', 'female': 'pt'},
        'ja': {'male': 'ja', 'female': 'ja'},
        'ko': {'male': 'ko', 'female': 'ko'},
        'zh': {'male': 'zh', 'female': 'zh'}
    },
    'rvc': {
        'male': {
            'en': 'models/rvc/male/en',
            'ru': 'models/rvc/male/ru',
            'es': 'models/rvc/male/es',
            'fr': 'models/rvc/male/fr',
            'de': 'models/rvc/male/de',
            'it': 'models/rvc/male/it',
            'pt': 'models/rvc/male/pt',
            'ja': 'models/rvc/male/ja',
            'ko': 'models/rvc/male/ko',
            'zh': 'models/rvc/male/zh'
        },
        'female': {
            'en': 'models/rvc/female/en',
            'ru': 'models/rvc/female/ru',
            'es': 'models/rvc/female/es',
            'fr': 'models/rvc/female/fr',
            'de': 'models/rvc/female/de',
            'it': 'models/rvc/female/it',
            'pt': 'models/rvc/female/pt',
            'ja': 'models/rvc/female/ja',
            'ko': 'models/rvc/female/ko',
            'zh': 'models/rvc/female/zh'
        }
    }
}

# === Model Configuration ===
MODEL_CONFIG = {
    'whisper': {
        'tiny': {'size': '75M', 'quality': 'low', 'speed': 'fast', 'cpu_ram': '1GB'},
        'base': {'size': '142M', 'quality': 'medium', 'speed': 'medium', 'cpu_ram': '1GB'},
        'small': {'size': '466M', 'quality': 'good', 'speed': 'medium', 'cpu_ram': '2GB'},
        'medium': {'size': '1.5B', 'quality': 'high', 'speed': 'slow', 'cpu_ram': '5GB'},
        'large': {'size': '2.9B', 'quality': 'best', 'speed': 'very slow', 'cpu_ram': '10GB'}
    },
    'translator': {
        'm2m100_418M': {
            'name': 'facebook/m2m100_418M',
            'quality': 'medium',
            'speed': 'fast'
        },
        'm2m100_1.2B': {
            'name': 'facebook/m2m100_1.2B',
            'quality': 'high',
            'speed': 'medium'
        },
        'nllb_200': {
            'name': 'facebook/nllb-200-distilled-600M',
            'quality': 'good',
            'speed': 'fast'
        },
        'nllb_600': {
            'name': 'facebook/nllb-200-1.3B',
            'quality': 'best',
            'speed': 'slow'
        }
    }
}

# === 1. Video transcription with Whisper ===
def transcribe_video(video_path, transcript_path, source_lang='en', model_size='base', use_gpu=False):
    print(f"🔍 Loading Whisper {model_size} model...")
    
    # Enhanced GPU detection and logging
    if use_gpu:
        if not torch.cuda.is_available():
            print("⚠️ GPU requested but CUDA is not available. Falling back to CPU.")
            device = "cpu"
        else:
            device = "cuda"
            print(f"💻 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        device = "cpu"
        print("💻 Using CPU for transcription")
    
    # Load model with appropriate settings
    try:
        model = whisper.load_model(
            model_size,
            device=device,
            download_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        )
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        if "CUDA" in str(e):
            print("⚠️ CUDA error detected. Falling back to CPU.")
            device = "cpu"
            model = whisper.load_model(
                model_size,
                device="cpu",
                download_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
            )
        else:
            raise e
    
    print("📼 Starting transcription...")
    
    # Configure transcription settings based on device
    transcribe_kwargs = {
        'language': LANGUAGE_CODES['whisper'][source_lang],
        'word_timestamps': True,
        'temperature': 0.0,  # Disable sampling for more consistent results
    }
    
    if device == "cpu":
        # CPU-optimized settings
        transcribe_kwargs.update({
            'fp16': False,  # Use full precision for CPU
            'beam_size': 3,  # Smaller beam size for CPU
            'best_of': 3,  # Fewer samples for CPU
            'condition_on_previous_text': False,  # Disable for faster CPU processing
        })
    else:
        # GPU settings
        transcribe_kwargs.update({
            'fp16': True,  # Use half precision for GPU
            'beam_size': 5,  # Larger beam size for GPU
            'best_of': 5,  # More samples for GPU
            'condition_on_previous_text': True,  # Enable for better quality
        })
    
    # Transcribe with configured settings
    result = model.transcribe(video_path, **transcribe_kwargs)
    
    # Process segments to create accurate chunks
    processed_segments = []
    current_segment = {
        'text': '',
        'start': None,
        'end': None,
        'words': []
    }
    
    # Maximum segment duration in seconds
    MAX_SEGMENT_DURATION = 5.0
    
    # Sentence ending punctuation
    sentence_endings = {'.', '!', '?', '...'}
    
    for segment in result['segments']:
        words = segment.get('words', [])
        if not words:
            continue
            
        # Process each word in the segment
        for word in words:
            # If this is a new segment
            if current_segment['start'] is None:
                current_segment = {
                    'text': word['word'],
                    'start': word['start'],
                    'end': word['end'],
                    'words': [word]
                }
            else:
                # Check if adding this word would exceed max duration
                would_exceed_duration = word['end'] - current_segment['start'] > MAX_SEGMENT_DURATION
                
                # Check if current segment ends with sentence ending
                current_text = current_segment['text'].strip()
                ends_with_sentence = any(current_text.endswith(end) for end in sentence_endings)
                
                if ends_with_sentence or would_exceed_duration:
                    # Save current segment and start a new one
                    processed_segments.append(current_segment)
                    current_segment = {
                        'text': word['word'],
                        'start': word['start'],
                        'end': word['end'],
                        'words': [word]
                    }
                else:
                    # Add word to current segment
                    current_segment['text'] += ' ' + word['word']
                    current_segment['end'] = word['end']
                    current_segment['words'].append(word)
    
    # Add the last segment if it exists
    if current_segment['text']:
        processed_segments.append(current_segment)
    
    # Write to file with proper formatting
    with open(transcript_path, "w", encoding="utf-8") as f:
        for seg in processed_segments:
            # Clean up the text
            text = seg['text'].strip()
            # Remove multiple spaces
            text = ' '.join(text.split())
            # Write with timing
            f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {text}\n")
    
    print(f"✅ Transcribed {len(processed_segments)} segments")
    return processed_segments

# === 2. Text translation ===
def translate_text(texts, output_path, source_lang='en', target_lang='ru', model_name='m2m100_418M'):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        raise ImportError("❌ Required packages not found. Install them: pip install transformers sentencepiece")

    print(f"🌐 Loading translation model: {model_name}...")
    model_config = MODEL_CONFIG['translator'][model_name]
    
    # Set custom cache directory to local folder
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📥 Downloading model to: {cache_dir}")
    
    # Check if model is already cached
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(model_config['name'], "config.json", cache_dir=cache_dir)
    if cached is None:
        print(f"⏳ First run: downloading {model_config['name']} (~430MB). This may take a few minutes...")
        print(f"   Please wait, the UI will update when download is complete.")
    else:
        print(f"✅ Model already cached, loading from disk...")

    # Download model
    snapshot_download(
        repo_id=model_config['name'],
        cache_dir=cache_dir,
        local_files_only=False,
    )
    
    print("🔄 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'], cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_config['name'], cache_dir=cache_dir)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("🚀 Using GPU for translation")
    
    print("🔁 Translating text...")
    translations = []
    batch_size = 4  # Adjust based on your GPU memory
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            # Set source and target languages
            tokenizer.src_lang = LANGUAGE_CODES['m2m100'][source_lang]
            tokenizer.tgt_lang = LANGUAGE_CODES['m2m100'][target_lang]
            
            # Tokenize the batch
            encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            # Generate translation with improved parameters
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(LANGUAGE_CODES['m2m100'][target_lang]),
                max_length=512,
                num_beams=5,
                length_penalty=0.6,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            # Decode the translations
            translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for orig, trans in zip(batch_texts, translated):
                print(f"Original ({SUPPORTED_LANGUAGES[source_lang]}): {orig[:50]}...")
                print(f"Translated ({SUPPORTED_LANGUAGES[target_lang]}): {trans[:50]}...")
                translations.append(trans)
            
        except Exception as e:
            print(f"⚠️ Translation error for batch {i//batch_size}: {str(e)}")
            print(f"Texts: {batch_texts}")
            translations.extend(batch_texts)  # Keep original text on error

    with open(output_path, "w", encoding="utf-8") as f:
        for line in translations:
            f.write(line + "\n")

    print(f"✅ Translated {len(translations)} segments")
    return translations

# === 3. Generate TTS with timing ===
def generate_tts_with_timing(texts, output_dir, base_name, segments, target_lang='ru', use_rvc=True, voice_gender='female', rvc_model=None):
    audio_files = []
    
    for i, (translated_text, seg) in enumerate(zip(texts, segments)):
        idx = f"{i:04d}"
        temp_raw = output_dir / f"{base_name}_{idx}_raw.mp3"
        temp_path = output_dir / f"{base_name}_{idx}.mp3"
        
        # Get exact duration from original segment
        segment_duration = float(seg['end']) - float(seg['start'])
        print(f"Segment {i}: Original duration = {segment_duration:.2f}s")
        
        # Generate TTS with voice gender
        generate_tts_audio(translated_text, str(temp_raw), target_lang, use_rvc, voice_gender, rvc_model)
        
        # Get actual TTS duration
        actual_duration = get_audio_duration(str(temp_raw))
        print(f"Segment {i}: TTS duration = {actual_duration:.2f}s")
        
        # Calculate required speed adjustment
        if actual_duration > segment_duration:
            speed = actual_duration / segment_duration
            if speed > 2.0:
                print(f"⚠️ Segment {i}: Audio too long ({actual_duration:.2f}s > {segment_duration:.2f}s), limited to 2x")
                speed = 2.0
            print(f"Segment {i}: Adjusting speed to {speed:.2f}x")
            
            # Adjust speed to match original duration
            command = [
                "ffmpeg", "-y", "-i", str(temp_raw),
                "-filter:a", f"atempo={speed:.3f}",
                str(temp_path)
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # If TTS is shorter, add silence to match original duration
            silence_duration = segment_duration - actual_duration
            print(f"Segment {i}: Adding {silence_duration:.2f}s silence")
            
            command = [
                "ffmpeg", "-y", "-i", str(temp_raw),
                "-af", f"apad=pad_dur={silence_duration}",
                str(temp_path)
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Verify final duration
        final_duration = get_audio_duration(str(temp_path))
        print(f"Segment {i}: Final duration = {final_duration:.2f}s (target: {segment_duration:.2f}s)")
        
        audio_files.append(str(temp_path))
    
    return audio_files

# === RVC Voice Conversion ===
class RVCConverter:
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🎵 Using device: {self.device}")
        
        # Load RVC model
        self.model_path = model_path
        self.model = self.load_rvc_model(model_path)
        
    def load_rvc_model(self, model_path):
        try:
            import torch
            import torchaudio
            import numpy as np
            from fairseq import checkpoint_utils
            import fairseq
            
            # Load the model files
            model_dir = os.path.dirname(model_path)
            index_path = os.path.join(model_dir, "added_drevnyirus_v2.index")
            pth_path = os.path.join(model_dir, "drevnyirus.pth")
            
            if not os.path.exists(index_path) or not os.path.exists(pth_path):
                raise FileNotFoundError(f"RVC model files not found in {model_dir}")
            
            # Load the model
            models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([pth_path])
            model = models[0].to(self.device)
            model.eval()
            
            print(f"✅ Loaded RVC model from {model_dir}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading RVC model: {e}")
            return None
        
    def convert_voice(self, audio_path, output_path):
        try:
            import torch
            import torchaudio
            import numpy as np
            
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            audio = audio.to(self.device)
            
            # Process with RVC
            with torch.no_grad():
                # Convert audio to the format expected by the model
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                
                # Apply the model
                converted_audio = self.model(audio)
                
                # Ensure the output is in the correct format
                if converted_audio.dim() == 1:
                    converted_audio = converted_audio.unsqueeze(0)
                
                # Save converted audio
                torchaudio.save(output_path, converted_audio.cpu(), sr)
                print(f"✅ Voice converted successfully: {output_path}")
                return True
                
        except Exception as e:
            print(f"⚠️ RVC conversion error: {e}")
            return False

# === 4. TTS with gTTS and RVC ===
def generate_tts_audio(text, output_path, lang='ru', use_rvc=True, voice_gender='female', rvc_model=None):
    try:
        # First try to use edge-tts for better voice quality and gender selection
        try:
            import edge_tts
            import asyncio
            
            async def generate_edge_tts():
                # Map language codes to Edge TTS voices
                voice_map = {
                    'ru': {'male': 'ru-RU-DmitryNeural', 'female': 'ru-RU-SvetlanaNeural'},
                    'en': {'male': 'en-US-GuyNeural', 'female': 'en-US-JennyNeural'},
                    # Spanish voices — change these to any voice from the list below:
                    # Female: es-ES-ElviraNeural, es-MX-DaliaNeural, es-AR-ElenaNeural, es-CO-SalomeNeural
                    # Male:   es-ES-AlvaroNeural, es-MX-JorgeNeural, es-AR-TomasNeural, es-CO-GonzaloNeural
                    'es': {'male': 'es-ES-AlvaroNeural', 'female': 'es-ES-ElviraNeural'},
                    'fr': {'male': 'fr-FR-DeniseNeural', 'female': 'fr-FR-HenriNeural'},
                    'de': {'male': 'de-DE-ConradNeural', 'female': 'de-DE-KatjaNeural'},
                    'it': {'male': 'it-IT-DiegoNeural', 'female': 'it-IT-ElsaNeural'},
                    'pt': {'male': 'pt-BR-AntonioNeural', 'female': 'pt-BR-FranciscaNeural'},
                    'ja': {'male': 'ja-JP-NanjoNeural', 'female': 'ja-JP-AiriNeural'},
                    'ko': {'male': 'ko-KR-InJoonNeural', 'female': 'ko-KR-SunHiNeural'},
                    'zh': {'male': 'zh-CN-YunxiNeural', 'female': 'zh-CN-XiaoxiaoNeural'}
                }
                
                voice = voice_map.get(lang, {}).get(voice_gender, f"{lang}-{voice_gender.upper()}-Neural")
                communicate = edge_tts.Communicate(text, voice)
                
                # Try to match original speech rate
                communicate.rate = "+0%"
                
                await communicate.save(output_path)
            
            asyncio.run(generate_edge_tts())
            print(f"✅ Generated audio with Edge TTS using {voice_gender} voice")
            return True
            
        except ImportError:
            print("⚠️ Edge TTS not installed. Installing now...")
            subprocess.run([sys.executable, "-m", "pip", "install", "edge-tts"])
            print("✅ Edge TTS installed. Please run the script again.")
            return False
        except Exception as e:
            print(f"⚠️ Edge TTS failed: {e}, falling back to gTTS")
        
        # Fallback to gTTS if edge-tts fails
        tts = gTTS(text, lang=VOICE_OPTIONS['gtts'][lang][voice_gender], slow=False)
        temp_path = output_path + ".temp.mp3"
        tts.save(temp_path)
        
        if use_rvc and rvc_model:
            # Convert with RVC
            rvc = RVCConverter(rvc_model)
            success = rvc.convert_voice(temp_path, output_path)
            if not success:
                print("⚠️ RVC conversion failed, using original TTS audio")
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
def main(video_path, source_lang='en', target_lang='ru', use_rvc=True, voice_gender='female', rvc_model=None, whisper_model='base', translator_model='m2m100_418M', use_gpu=False):
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
        segments = transcribe_video(video_path, transcript_file, source_lang, whisper_model, use_gpu)

    texts = [seg['text'] for seg in segments]

    # === Translation ===
    if translated_file.exists():
        print(f"⏩ Skipping translation — found file {translated_file}")
        with open(translated_file, "r", encoding="utf-8") as f:
            translations = [line.strip() for line in f.readlines()]
    else:
        print("🌐 Translating text...")
        translations = translate_text(texts, translated_file, source_lang, target_lang, translator_model)

    # === TTS with timing ===
    expected_audio_files = [str(tts_chunks_dir / f"{base_name}_{i:04d}.mp3") for i in range(len(translations))]
    if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in expected_audio_files):
        print("⏩ Skipping TTS audio generation — all audio files found in folder.")
        temp_audio_files = expected_audio_files
    else:
        print(f"🔊 Generating TTS audio with {voice_gender} voice...")
        temp_audio_files = generate_tts_with_timing(
            translations, 
            tts_chunks_dir, 
            base_name, 
            segments,
            target_lang,
            use_rvc,
            voice_gender,
            rvc_model
        )
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
    parser.add_argument('--voice-gender', '-g', choices=['male', 'female'], default='female',
                      help='Voice gender (default: female)')
    parser.add_argument('--rvc-model', '-m', help='Path to custom RVC model (optional)')
    parser.add_argument('--whisper-model', '-w', choices=MODEL_CONFIG['whisper'].keys(), default='base',
                      help='Whisper model size (default: base)')
    parser.add_argument('--translator-model', '-tr', choices=MODEL_CONFIG['translator'].keys(), default='m2m100_418M',
                      help='Translation model (default: m2m100_418M)')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for Whisper transcription')
    args = parser.parse_args()
    
    print(f"🌍 Translating from {SUPPORTED_LANGUAGES[args.source_lang]} to {SUPPORTED_LANGUAGES[args.target_lang]}")
    print(f"🎤 Using Whisper {args.whisper_model} model")
    print(f"🌐 Using {args.translator_model} for translation")
    
    # Get RVC model path
    rvc_model = None
    if not args.no_rvc:
        if args.rvc_model:
            rvc_model = args.rvc_model
        else:
            rvc_model = VOICE_OPTIONS['rvc'][args.voice_gender][args.target_lang]
            if not os.path.exists(rvc_model):
                print(f"⚠️ RVC model not found at {rvc_model}, falling back to Edge TTS")
                args.no_rvc = True
    
    if args.no_rvc:
        print(f"🎤 Using Edge TTS with {args.voice_gender} voice")
    else:
        print(f"🎤 Using RVC with {args.voice_gender} voice")
    
    main(args.video_path, args.source_lang, args.target_lang, not args.no_rvc, args.voice_gender, rvc_model, args.whisper_model, args.translator_model, args.use_gpu)
