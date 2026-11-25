import json
import subprocess
import sys
from html import escape
from pathlib import Path
from subprocess import TimeoutExpired
from tempfile import TemporaryDirectory

import streamlit as st
import streamlit.components.v1 as components

from signspeak.media import dictionary, map_signs, render_clips
from signspeak.settings import CHECKPOINT, FULL_MODEL, IMAGE_TYPES, VIDEO_TYPES, ROOT

st.set_page_config(page_title="SignSpeak", page_icon="🤟", layout="centered")
st.title("SignSpeak")
listener, communicator = st.tabs(["Listener · ISL to English", "Communicator · English to ISL"])


def speech_player(text):
    components.html(f"""
        <button id="speak" type="button" data-text="{escape(text, quote=True)}">Speak English</button>
        <span id="status" role="status"></span>
        <script>
          const button = document.getElementById("speak");
          const status = document.getElementById("status");
          button.onclick = () => {{
            if (!window.speechSynthesis) {{
              status.textContent = "Speech synthesis is unavailable in this browser.";
              return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(button.dataset.text);
            utterance.lang = "en-US";
            utterance.onstart = () => status.textContent = "Speaking…";
            utterance.onend = () => status.textContent = "";
            utterance.onerror = () => status.textContent = "Speech playback failed.";
            window.speechSynthesis.speak(utterance);
          }};
        </script>
    """, height=48)


with listener:
    full = FULL_MODEL.is_file()
    checkpoint = FULL_MODEL if full else CHECKPOINT
    if full:
        st.caption("INCLUDE recognition · upload one isolated sign per file. Low-confidence matches appear as suggestions.")
    st.write("Upload one image or one trimmed video per sign. For a sequence, select files in signing order.")
    uploads = st.file_uploader("Sign images or clips", type=IMAGE_TYPES + VIDEO_TYPES,
                               accept_multiple_files=True)
    if uploads:
        st.caption("Playback order · " + " → ".join(upload.name for upload in uploads))
    read_aloud = st.checkbox("Generate English speech")
    if st.button("Recognize signs", type="primary", disabled=not uploads):
        if len(uploads) > 12:
            st.error("Upload up to 12 signs at a time.")
        elif not checkpoint.is_file():
            st.error("The recognition checkpoint is missing. Run python -m signspeak.train first.")
        else:
            try:
                output = []
                with st.spinner("Recognizing signs on GPU…"), TemporaryDirectory() as directory:
                    paths = []
                    for index, upload in enumerate(uploads, 1):
                        path = Path(directory) / f"{index}{Path(upload.name).suffix.lower()}"
                        path.write_bytes(upload.getbuffer())
                        paths.append(str(path))
                    run = subprocess.run(
                        [sys.executable, "-m", "signspeak.worker", *paths], cwd=ROOT.parent,
                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                        timeout=60 + 90 * len(paths),
                    )
                    if run.returncode:
                        print(run.stderr, file=sys.stderr, flush=True)
                        raise RuntimeError(f"Recognition stopped (worker exit {run.returncode}). Try one shorter clip.")
                    for index, (upload, item) in enumerate(zip(uploads, json.loads(run.stdout), strict=True), 1):
                        if "error" in item:
                            st.warning(f"{upload.name} · {item['error']}")
                            output.append(None)
                            continue
                        sign, score = item["sign"], item["score"]
                        if item["uncertain"]:
                            st.warning(f"{index}. Uncertain · closest match: {sign} ({score:.0%})")
                            output.append(None)
                            continue
                        output.append(sign)
                        st.write(f"{index}. {sign} · model score {score:.0%}")
                    if any(output):
                        result = " · ".join(sign or "[unrecognized]" for sign in output)
                        st.subheader("English")
                        st.write(result)
                        if read_aloud:
                            speech_player(". ".join(sign or "unrecognized" for sign in output))
            except (OSError, ValueError, RuntimeError, TimeoutExpired) as error:
                st.error(str(error))
    st.caption("Image predictions use a single pose. For signs with movement, upload a video.")

with communicator:
    st.write("Enter a supported sign or a sequence of supported signs.")
    try:
        entries = dictionary()
    except (OSError, KeyError, ValueError) as error:
        entries = {}
        st.error(f"Could not load the sign dictionary. {error}")
    available = {key: (label, path) for key, (label, path) in entries.items() if path.is_file()}
    if available:
        with st.expander(f"Browse {len(available)} supported words and phrases"):
            query = st.text_input("Find a sign or sentence")
            st.write(" · ".join(sorted(label for label, _ in available.values()
                                       if query.casefold() in label.casefold())))
    elif entries:
        st.warning("The dictionary clips are missing from the dataset folder.")
    message = st.text_area("English text", placeholder="Hello, good morning", max_chars=240)
    if st.button("Play signs", type="primary", disabled=not (available and message.strip())):
        matches = list(map_signs(message, available))
        clips = [(label, path) for label, path in matches if path is not None]
        missing = [label for label, path in matches if path is None]
        if not clips:
            st.info("No matching sign clips. Try one of the available signs above.")
        elif len(clips) > 12:
            st.error("Enter up to 12 supported signs at a time.")
        else:
            try:
                with st.spinner("Preparing sign clips…"), TemporaryDirectory() as directory:
                    output = Path(directory) / "signs.mp4"
                    if missing:
                        for label, path in matches:
                            if path is None:
                                st.warning(f"No matching clip for “{label}”.")
                            else:
                                st.write(label)
                                st.video(render_clips([path], output))
                    else:
                        st.write(" → ".join(label for label, _ in clips))
                        st.video(render_clips([path for _, path in clips], output))
            except (OSError, ValueError, RuntimeError, TimeoutExpired) as error:
                st.error(str(error))
    st.caption("Clips play in the order entered. Use the vocabulary above for word or phrase playback.")
