import base64, io, json, os, re
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment
import engine as E

st.set_page_config(page_title="Audio Revision Studio", layout="wide")
S = st.session_state
player = components.declare_component(
    "rev_player", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_component"))

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("Settings")
    key = st.text_input("Gemini API key", os.getenv("GEMINI_API_KEY", ""), type="password")
    text_model = st.text_input("Text model (analysis / rewrite)", "gemini-3.1-pro-preview")
    tts_model = st.text_input("TTS model", "gemini-2.5-pro-preview-tts")
    v = st.selectbox("Voice", E.VOICES + ["Other…"])
    voice = st.text_input("Custom voice name", "") if v == "Other…" else v
    style = st.text_area("Style prompt", "Speak in natural Egyptian Arabic, calm and clear, "
                         "same pace and tone as the surrounding narration.")
    dialect = st.text_input("Dialect", "Egyptian Arabic")
    tries = st.slider("Max auto-retries for tone match", 1, 5, 3)
    if st.button("↩ Undo last change", disabled=not S.get("history")):
        S.audio, S.manifest = S.history.pop()
        S.cand = None
        S.ver += 1
        st.rerun()

st.title("🎧 Audio Revision Studio")

# ---------------- Upload ----------------
c1, c2 = st.columns(2)
af = c1.file_uploader("1. Audio file", type=["mp3", "wav", "m4a", "ogg"])
mf = c2.file_uploader("2a. Manifest JSON (if you have one)", type=["json"])
sf = c2.file_uploader("2b. …or the script text file (.txt)", type=["txt"])
if not af:
    st.info("Upload the audio to begin.")
    st.stop()

sig = (af.name, af.size, mf and mf.size, sf and sf.size)
if S.get("sig") != sig:
    S.sig, S.history, S.cand, S.out, S.draft = sig, [], None, None, None
    S.fmt = af.name.rsplit(".", 1)[-1].lower()
    S.audio = AudioSegment.from_file(af)
    S.ver = S.get("ver", 0) + 1
    S.manifest = E.load_manifest(mf.getvalue()) if mf else None

if S.manifest is None:
    if not sf:
        st.info("Upload a manifest JSON or the script text file.")
        st.stop()
    sentences = E.split_script(sf.getvalue().decode("utf-8"))
    st.subheader("Build sentence timings from the script")
    st.caption(f"{len(sentences)} sentences found (split by line breaks and . ! ? ؟)")
    if st.button("⚙ Align script to audio", disabled=not key):
        bar = st.progress(0.0, text="Starting…")
        try:
            S.draft, S.draft_msg = E.build_manifest(E.client(key), text_model, S.audio,
                                                    sentences, progress=lambda f, t: bar.progress(f, text=t))
        except Exception as e:
            st.error(f"Alignment failed: {e}")
    if S.get("draft"):
        if S.get("draft_msg"):
            st.warning(S.draft_msg)
        st.write("Check the times (ms) and fix any that are wrong, then confirm.")
        ed = st.data_editor(S.draft, num_rows="fixed", use_container_width=True)
        if st.button("✅ Use this manifest"):
            S.manifest = [dict(id=int(r["id"]), text=r["text"],
                               start_ms=int(r["start_ms"]), end_ms=int(r["end_ms"])) for r in ed]
            st.rerun()
    st.stop()

audio, manifest = S.audio, S.manifest

if S.get("b64_ver") != S.ver:  # light mp3 copy for the in-browser player
    buf = io.BytesIO()
    audio.set_channels(1).export(buf, format="mp3", bitrate="96k")
    S.b64 = "data:audio/mpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    S.b64_ver = S.ver

left, right = st.columns([3, 2], gap="large")
with left:
    sel = player(audio=S.b64, ver=S.ver, key="player", default=None,
                 segments=[dict(id=m["id"], text=m["text"], start_ms=m["start_ms"],
                                end_ms=m["end_ms"]) for m in manifest])

with right:
    st.subheader("Fix a mistake")
    live = sel and sel.get("n", 0) > S.get("cleared_n", 0)
    seg = next((m for m in manifest if m["id"] == sel["sid"]), None) if live else None
    if not seg:
        st.info("Press play and listen. When you hear a mispronounced word, **click it** in the "
                "transcript, then press Regenerate here.")
    else:
        word = re.sub(r"[\s.,!?؟،:;\"'()«»\[\]-]+", "", sel.get("word") or "") or None
        wi = sel.get("wi")
        clip = audio[seg["start_ms"]:seg["end_ms"]]
        st.markdown(f"**Sentence #{seg['id'] + 1}**" + (f" — word: **{word}**" if word else
                    " — whole sentence (Gemini will listen for the problem)"))
        st.audio(E.wav_bytes(clip), format="audio/wav")
        corr = st.text_input("Correct pronunciation (optional — leave empty to let Gemini decide)")
        if st.button("🔁 Regenerate", type="primary", disabled=not key):
            try:
                c = E.client(key)
                with st.spinner("Rewriting the sentence…"):
                    d = E.diagnose(c, text_model, clip, seg.get("tts_text", seg["text"]),
                                   dialect, word, wi, corr.strip() or None)
                with st.spinner("Generating audio and checking tone…"):
                    new, m = E.regenerate(c, tts_model, voice, d["text"], style, audio,
                                          manifest, seg, tries)
                S.cand = dict(key=(seg["id"], wi), new=new, m=m, text=d["text"],
                              word=d.get("word", word))
            except Exception as ex:
                st.error(f"Failed: {ex}")
        cd = S.get("cand")
        if cd and cd["key"] == (seg["id"], wi):
            p1, p2 = st.columns(2)
            p1.write("Original"); p1.audio(E.wav_bytes(clip), format="audio/wav")
            p2.write("Regenerated"); p2.audio(E.wav_bytes(cd["new"]), format="audio/wav")
            st.caption(f"Changed word: {cd['word']}  |  TTS text: {cd['text']}")
            m = cd["m"]
            if m["ok"]:
                st.success(f"Tone matches neighbours — pitch Δ {m.get('pitch_semitones', 0)} st, "
                           f"pace Δ {m.get('pace_log2', 0)}")
            else:
                st.warning(f"Tone is off — pitch Δ {m['pitch_semitones']} st, pace Δ "
                           f"{m['pace_log2']}. Regenerate again or change voice/style.")
            k1, k2 = st.columns(2)
            if k1.button("✅ Accept & insert"):
                S.history.append((audio, manifest))
                S.audio, S.manifest = E.splice(audio, manifest, seg, cd["new"], cd["text"])
                S.cand, S.out, S.cleared_n = None, None, sel["n"]
                S.ver += 1
                st.rerun()
            if k2.button("🗑 Discard"):
                S.cand = None
                st.rerun()

# ---------------- Export ----------------
st.divider()
if st.button("📦 Prepare download"):
    buf = io.BytesIO()
    audio.export(buf, format=S.fmt)
    S.out = buf.getvalue()
if S.get("out"):
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Revised audio", S.out, f"revised.{S.fmt}")
    d2.download_button("⬇ Updated manifest", json.dumps(manifest, ensure_ascii=False, indent=1),
                       "manifest_revised.json")
