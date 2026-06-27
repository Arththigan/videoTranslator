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
    output_folder,
    force_regen,
    progress=gr.Progress(track_tqdm=True),
):
    if video_file is None:
        yield None, "⚠️ Please upload a video file first.", gr.DownloadButton(visible=False)
        return

    source_lang = code_from_choice(source_lang_str)
    target_lang = code_from_choice(target_lang_str)
    custom_out  = output_folder.strip() if output_folder and output_folder.strip() else ""

    # Auto-detect already-processed video — force regenerate so settings are applied fresh
    base     = sanitize_filename(Path(video_file).stem)
    out_dir  = Path("output") / base
    final_output = out_dir / f"{base}_dubbed.mp4"
    if final_output.exists() and not force_regen:
        force_regen = True
        print(f"🔄 Previously dubbed video detected, auto-regenerating with current settings...")

    # If force regenerate — delete TTS chunks, dubbed audio and final video so they rebuild
    if force_regen and video_file:
        base     = sanitize_filename(Path(video_file).stem)
        out_dir  = Path("output") / base
        if out_dir.exists():
            import stat

            def _force_remove(func, path, _):
                """Handle read-only files on Windows."""
                os.chmod(path, stat.S_IWRITE)
                func(path)

            tts_dir = out_dir / "tts-chunks"
            if tts_dir.exists():
                shutil.rmtree(str(tts_dir), onerror=_force_remove)
            for fname in ["audio_dubbed.mp3", f"{base}_dubbed.mp4"]:
                fp = out_dir / fname
                if fp.exists():
                    try:
                        os.chmod(str(fp), stat.S_IWRITE)
                        fp.unlink()
                    except Exception:
                        pass
            if force_regen:
                print(f"🗑️ Cleared previous TTS/audio/video cache for: {base}")

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
                force_regen      = force_regen,
            )
            base     = sanitize_filename(Path(video_file).stem)
            out_file = Path("output") / base / f"{base}_dubbed.mp4"

            if out_file.exists():
                if custom_out:
                    # Copy to the user-chosen folder
                    dest_dir = Path(custom_out)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / out_file.name
                    shutil.copy2(str(out_file), str(dest_file))
                    print(f"📂 Copied to: {dest_file}")
                    result_path[0] = str(dest_file)
                else:
                    result_path[0] = str(out_file)
        except Exception as e:
            import traceback
            error_box[0] = f"{e}\n\nFull traceback:\n{traceback.format_exc()}"

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
        logs.append(f"\n🎉 Done! Saved to: {result_path[0]}")
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


# ── History helpers ───────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")

def get_dubbed_videos():
    """Scan output/ folder and return list of all dubbed mp4 files, newest first."""
    if not OUTPUT_DIR.exists():
        return []
    files = sorted(
        OUTPUT_DIR.rglob("*_dubbed.mp4"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return files

def build_history_html():
    """Build a Chrome download history style list — no video embeds, instant load."""
    files = get_dubbed_videos()
    if not files:
        return "<div style='padding:30px;text-align:center;color:#888;font-size:16px;'>No dubbed videos yet. Translate a video to see it here.</div>"

    rows = []
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        mtime   = __import__("datetime").datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d  %H:%M")
        name    = f.name
        folder  = str(f.parent)
        dl_url  = f"/gradio_api/file={f.resolve()}"

        rows.append(f"""
        <div style="
            display:flex; align-items:center; justify-content:space-between;
            padding:12px 16px; border-bottom:1px solid #e5e7eb;
            background:#fff; gap:12px;
        " onmouseover="this.style.background='#f9fafb'" onmouseout="this.style.background='#fff'">

            <!-- Icon + file info -->
            <div style="display:flex;align-items:center;gap:12px;min-width:0;flex:1;">
                <div style="font-size:28px;flex-shrink:0;">🎬</div>
                <div style="min-width:0;">
                    <div style="font-weight:600;font-size:14px;color:#111;
                                overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                         title="{name}">
                        {name}
                    </div>
                    <div style="font-size:12px;color:#6b7280;margin-top:2px;">
                        {mtime} &nbsp;·&nbsp; {size_mb:.1f} MB &nbsp;·&nbsp;
                        <span title="{folder}" style="cursor:default;">{folder[:60]}{'...' if len(folder)>60 else ''}</span>
                    </div>
                </div>
            </div>

            <!-- Action buttons -->
            <div style="display:flex;gap:8px;flex-shrink:0;">
                <a href="{dl_url}" download="{name}"
                   style="padding:6px 14px;background:#16a34a;color:#fff;border-radius:6px;
                          text-decoration:none;font-size:13px;font-weight:600;white-space:nowrap;">
                    ⬇️ Download
                </a>
                <a href="{dl_url}" target="_blank"
                   style="padding:6px 14px;background:#2563eb;color:#fff;border-radius:6px;
                          text-decoration:none;font-size:13px;font-weight:600;white-space:nowrap;">
                    ▶ Play
                </a>
            </div>
        </div>
        """)

    rows_html = "\n".join(rows)
    count = len(files)
    return f"""
    <div style="border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;background:#fff;">
        <div style="padding:12px 16px;background:#f3f4f6;border-bottom:1px solid #e5e7eb;
                    font-size:13px;color:#6b7280;font-weight:600;">
            {count} dubbed video{"s" if count != 1 else ""}
        </div>
        {rows_html}
    </div>
    """

def refresh_history():
    return build_history_html()


# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
#title { text-align: center; }
/* Auto-scroll progress log to bottom */
#log_box textarea {
    font-family: monospace;
    font-size: 12px;
    overflow-y: auto !important;
    scroll-behavior: smooth;
}
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

    # Auto-scroll the progress log textarea to the bottom whenever it updates
    gr.HTML("""
    <script>
    function setupLogScroll() {
        const observer = new MutationObserver(() => {
            const el = document.querySelector('#log_box textarea');
            if (el) { el.scrollTop = el.scrollHeight; }
        });
        const target = document.querySelector('#log_box');
        if (target) {
            observer.observe(target, { childList: true, subtree: true, characterData: true });
        } else {
            setTimeout(setupLogScroll, 500);
        }
    }
    document.addEventListener('DOMContentLoaded', setupLogScroll);
    setTimeout(setupLogScroll, 1000);
    </script>
    """)

    # Auto-scroll the progress log textarea to the bottom whenever it updates
    gr.HTML("""
    <script>
    function setupLogScroll() {
        const observer = new MutationObserver(() => {
            const el = document.querySelector('#log_box textarea');
            if (el) { el.scrollTop = el.scrollHeight; }
        });
        const target = document.querySelector('#log_box');
        if (target) {
            observer.observe(target, { childList: true, subtree: true, characterData: true });
        } else {
            setTimeout(setupLogScroll, 500);
        }
    }
    document.addEventListener('DOMContentLoaded', setupLogScroll);
    setTimeout(setupLogScroll, 1000);
    </script>
    """)

    gr.Markdown("# 🎬 Video Translator", elem_id="title")
    gr.Markdown("Dub any video into another language using AI — powered by Whisper + M2M100 + Edge TTS.")

    with gr.Tabs():

        # ════════════════════════════════════════════
        # TAB 1 — Translate
        # ════════════════════════════════════════════
        with gr.TabItem("🚀 Translate"):
            with gr.Row():
                # ── LEFT COLUMN ──────────────────────────────────────────────
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

                    force_regen = gr.Checkbox(
                        label="🔄 Force Regenerate — re-dub even if already processed (use when changing voice/language)",
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
                        output_folder = gr.Textbox(
                            label="📂 Save Output To (optional)",
                            placeholder=r"e.g. D:\DubbedVideos  or  E:\MyVideos  — leave blank to use default output\ folder",
                            info="Type any folder path on your PC or external drive. Folder will be created if it doesn't exist.",
                            value="",
                        )

                    translate_btn = gr.Button("🚀 Translate Video", variant="primary", size="lg")

                # ── RIGHT COLUMN ─────────────────────────────────────────────
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

            # ── language info box ────────────────────────────────────────────
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

            # ── wire up translate button ─────────────────────────────────────
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
                    output_folder,
                    force_regen,
                ],
                outputs=[video_output, log_output, export_btn],
            )

        # ════════════════════════════════════════════
        # TAB 2 — Dubbed History
        # ════════════════════════════════════════════
        with gr.TabItem("📼 Dubbed History") as history_tab:
            with gr.Row():
                gr.Markdown("### All dubbed videos — click play or download any file")
                refresh_btn = gr.Button("🔄 Refresh", scale=0, min_width=120)

            history_html = gr.HTML(value=build_history_html)

            refresh_btn.click(fn=refresh_history, outputs=history_html)

            # Also refresh history when a new video is dubbed
            translate_btn.click(fn=refresh_history, outputs=history_html)


# ── Auto-cleanup: delete output folders older than 7 days ────────────────────
def cleanup_old_outputs(max_age_days: int = 7):
    """Delete any output subfolder whose dubbed mp4 is older than max_age_days."""
    import time
    cutoff = time.time() - max_age_days * 86400  # 86400 seconds in a day
    if not OUTPUT_DIR.exists():
        return
    deleted = []
    for folder in OUTPUT_DIR.iterdir():
        if not folder.is_dir():
            continue
        # Use the dubbed mp4 mtime if it exists, otherwise the folder mtime
        mp4_files = list(folder.glob("*_dubbed.mp4"))
        ref_time = mp4_files[0].stat().st_mtime if mp4_files else folder.stat().st_mtime
        if ref_time < cutoff:
            try:
                import stat as _stat
                def _force_remove(func, path, _):
                    os.chmod(path, _stat.S_IWRITE)
                    func(path)
                shutil.rmtree(str(folder), onerror=_force_remove)
                deleted.append(folder.name)
            except Exception as e:
                print(f"⚠️ Could not delete {folder}: {e}")
    if deleted:
        print(f"🗑️ Auto-cleanup: deleted {len(deleted)} output(s) older than {max_age_days} days: {', '.join(deleted)}")


# ── launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cleanup_old_outputs(max_age_days=7)   # run once on startup
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False,
        theme=gr.themes.Soft(),
        css=css,
        allowed_paths=[str(Path("output").resolve())],
    )
