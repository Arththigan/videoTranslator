import sys
import os
import threading
import queue
import shutil
from pathlib import Path

# ── Fix uvicorn logging incompatibility with Python 3.14 ─────────────────────
# Python 3.14 removed support for custom formatter factories via "()" key.
# Patch uvicorn's logging config to use plain formatters before importing gradio.
import uvicorn.config as _uvc
_uvc.LOGGING_CONFIG["formatters"]["default"] = {
    "format": "%(levelname)s %(message)s",
}
_uvc.LOGGING_CONFIG["formatters"]["access"] = {
    "format": "%(levelname)s %(message)s",
}

import gradio as gr

# ── redirect print() so the UI can capture logs ──────────────────────────────
_log_queue = queue.Queue()

class _QueueWriter:
    def __init__(self, original):
        self._orig = original
    def write(self, msg):
        if msg.strip():
            _log_queue.put(msg.rstrip())
        self._orig.write(msg)
    def flush(self):
        self._orig.flush()

sys.stdout = _QueueWriter(sys.__stdout__)
sys.stderr = _QueueWriter(sys.__stderr__)

# ── import app logic ──────────────────────────────────────────────────────────
from main import (
    main as run_translation,
    SUPPORTED_LANGUAGES,
    MODEL_CONFIG,
    sanitize_filename,
)

# ── constants ─────────────────────────────────────────────────────────────────
LANG_CHOICES   = [f"{name} ({code})" for code, name in SUPPORTED_LANGUAGES.items()]
WHISPER_MODELS = list(MODEL_CONFIG["whisper"].keys())
TRANS_MODELS   = list(MODEL_CONFIG["translator"].keys())

def code_from_choice(choice: str) -> str:
    """Extract language code from 'English (en)' style string."""
    return choice.split("(")[-1].rstrip(")")

# ── core translate function called by Gradio ──────────────────────────────────
def translate_video(
    video_file,
    source_lang_str,
    target_lang_str,
    voice_gender,
    whisper_model,
    translator_model,
    use_gpu,
    progress=gr.Progress(track_tqdm=True),
):
    if video_file is None:
        yield None, "⚠️ Please upload a video file first.", gr.DownloadButton(visible=False)
        return

    source_lang = code_from_choice(source_lang_str)
    target_lang = code_from_choice(target_lang_str)

    # collect logs
    logs = []
    result_path = [None]
    error_box   = [None]

    def _run():
        try:
            run_translation(
                video_path       = video_file,
                source_lang      = source_lang,
                target_lang      = target_lang,
                use_rvc          = False,
                voice_gender     = voice_gender,
                rvc_model        = None,
                whisper_model    = whisper_model,
                translator_model = translator_model,
                use_gpu          = use_gpu,
            )
            base      = sanitize_filename(Path(video_file).stem)
            out_file  = Path("output") / base / f"{base}_dubbed.mp4"
            result_path[0] = str(out_file) if out_file.exists() else None
        except Exception as e:
            error_box[0] = str(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # stream logs back to UI while thread is running
    heartbeat = 0
    while thread.is_alive():
        new_lines = []
        while not _log_queue.empty():
            new_lines.append(_log_queue.get_nowait())
        if new_lines:
            logs.extend(new_lines)
            heartbeat = 0
            yield None, "\n".join(logs), gr.DownloadButton(visible=False)
        else:
            heartbeat += 1
            if heartbeat % 10 == 0:
                mins = (heartbeat // 10) * 5 // 60
                secs = (heartbeat // 10) * 5 % 60
                waiting_msg = f"⏳ Still working... ({mins:02d}:{secs:02d} elapsed) — downloading model or processing, please wait."
                yield None, "\n".join(logs) + f"\n{waiting_msg}", gr.DownloadButton(visible=False)
        threading.Event().wait(0.5)

    # drain any remaining log lines
    while not _log_queue.empty():
        logs.append(_log_queue.get_nowait())

    if error_box[0]:
        logs.append(f"\n❌ Error: {error_box[0]}")
        yield None, "\n".join(logs), gr.DownloadButton(visible=False)
        return

    if result_path[0]:
        logs.append("\n🎉 Done! Your dubbed video is ready to download.")
        yield (
            result_path[0],
            "\n".join(logs),
            gr.DownloadButton(
                value=result_path[0],
                label="⬇️ Export / Download Dubbed Video",
                visible=True,
            ),
        )
    else:
        logs.append("\n❌ Output file not found. Check the log above for errors.")
        yield None, "\n".join(logs), gr.DownloadButton(visible=False)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
#title { text-align: center; }
#log_box textarea { font-family: monospace; font-size: 12px; }
footer { display: none !important; }

/* Fix video components to a fixed height — never grow beyond this */
#video_input, #video_output {
    height: 280px !important;
    max-height: 280px !important;
}
#video_input video, #video_output video {
    height: 240px !important;
    max-height: 240px !important;
    width: 100% !important;
    object-fit: contain !important;
}
#video_input .wrap, #video_output .wrap {
    height: 240px !important;
    max-height: 240px !important;
}

/* Export button — full width, green when visible */
#export_btn {
    width: 100% !important;
    background: #16a34a !important;
    color: white !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
#export_btn:hover {
    background: #15803d !important;
}
"""

with gr.Blocks(title="🎬 Video Translator") as demo:

    gr.Markdown("# 🎬 Video Translator", elem_id="title")
    gr.Markdown("Dub any video into another language using AI — powered by Whisper + M2M100 + Edge TTS.")

    with gr.Row():
        # ── LEFT COLUMN ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="📁 Upload Video",
                sources=["upload"],
                elem_id="video_input",
            )

            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=LANG_CHOICES,
                    value="English (en)",
                    label="🗣️ Source Language",
                )
                target_lang = gr.Dropdown(
                    choices=LANG_CHOICES,
                    value="Spanish (es)",
                    label="🌍 Target Language",
                )

            with gr.Row():
                voice_gender = gr.Radio(
                    choices=["female", "male"],
                    value="female",
                    label="🎤 Voice Gender",
                )
                use_gpu = gr.Checkbox(
                    label="⚡ Use GPU (if available)",
                    value=False,
                )

            with gr.Accordion("⚙️ Advanced Options", open=False):
                whisper_model = gr.Dropdown(
                    choices=WHISPER_MODELS,
                    value="base",
                    label="🔍 Whisper Model (transcription quality)",
                    info="tiny=fastest  base=default  small=better  medium=high  large=best",
                )
                translator_model = gr.Dropdown(
                    choices=TRANS_MODELS,
                    value="m2m100_418M",
                    label="🌐 Translation Model",
                    info="418M=fast  1.2B=better quality  nllb=alternative",
                )

            translate_btn = gr.Button("🚀 Translate Video", variant="primary", size="lg")

        # ── RIGHT COLUMN ─────────────────────────────────────────────────────
        with gr.Column(scale=1):
            video_output = gr.Video(
                label="✅ Dubbed Video",
                interactive=False,
                elem_id="video_output",
            )
            export_btn = gr.DownloadButton(
                label="⬇️ Export / Download Dubbed Video",
                variant="secondary",
                size="lg",
                visible=False,
                elem_id="export_btn",
            )
            log_output = gr.Textbox(
                label="📋 Progress Log",
                lines=15,
                max_lines=15,
                interactive=False,
                elem_id="log_box",
                placeholder="Logs will appear here once you start translation...",
            )

    # ── language info box ────────────────────────────────────────────────────
    with gr.Accordion("📖 Supported Languages & Voice Reference", open=False):
        gr.Markdown("""
| Code | Language | Male Voice | Female Voice |
|------|----------|-----------|--------------|
| en | English | en-US-GuyNeural | en-US-JennyNeural |
| es | Spanish | es-ES-AlvaroNeural | es-ES-ElviraNeural |
| ru | Russian | ru-RU-DmitryNeural | ru-RU-SvetlanaNeural |
| fr | French | fr-FR-HenriNeural | fr-FR-DeniseNeural |
| de | German | de-DE-ConradNeural | de-DE-KatjaNeural |
| it | Italian | it-IT-DiegoNeural | it-IT-ElsaNeural |
| pt | Portuguese | pt-BR-AntonioNeural | pt-BR-FranciscaNeural |
| ja | Japanese | ja-JP-NanjoNeural | ja-JP-AiriNeural |
| ko | Korean | ko-KR-InJoonNeural | ko-KR-SunHiNeural |
| zh | Chinese | zh-CN-YunxiNeural | zh-CN-XiaoxiaoNeural |
        """)

    # ── wire up button ───────────────────────────────────────────────────────
    translate_btn.click(
        fn=translate_video,
        inputs=[
            video_input,
            source_lang,
            target_lang,
            voice_gender,
            whisper_model,
            translator_model,
            use_gpu,
        ],
        outputs=[video_output, log_output, export_btn],
    )

# ── launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False,
        theme=gr.themes.Soft(),
        css=css,
    )
