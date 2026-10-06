# Audio Revision Studio

## Setup (once, by whoever hosts it)
1. Install Python 3.10+ and ffmpeg (`sudo apt install ffmpeg` / `brew install ffmpeg`).
2. `pip install -r requirements.txt`
3. Set the key: `export GEMINI_API_KEY=your_key` (or paste it in the sidebar).
4. Run: `streamlit run app.py` -> open the URL shown.
   To host as a website: push this folder to GitHub and deploy on Streamlit Community Cloud
   (`packages.txt` installs ffmpeg; add GEMINI_API_KEY under Secrets/env).

## Manifest format (your builder must write this next to every audio file)
[{"id":0,"text":"full sentence","start_ms":0,"end_ms":5200,"voice":"Charon"}, ...]
Milliseconds, in the order of the final audio.

## Reviser steps
1. Upload the audio plus EITHER a manifest JSON OR the script .txt. With the script: press
   Align script to audio, check/fix the times in the table, press Use this manifest.
2. In the sidebar check the API key, models, voice and style prompt.
3. Press play on the player (waveform on top, transcript below follows the audio).
   Click the waveform to jump anywhere.
4. When you hear a mispronounced word, click that word in the transcript (orange).
   Double-click a word, or click the ▶ number, to replay that sentence.
5. On the right press Regenerate. Optionally type the correct pronunciation first.
6. Compare original vs regenerated and read the tone message. Not happy -> Regenerate again.
7. Press Accept & insert. The audio and transcript update in place; Undo is in the sidebar.
8. Keep listening, fix more, then Prepare download for the revised audio + manifest.
