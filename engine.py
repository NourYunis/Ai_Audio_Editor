import copy, io, json, re
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, detect_silence
from google import genai
from google.genai import types

VOICES = ["Charon", "Puck", "Kore", "Zephyr", "Fenrir", "Leda", "Orus", "Aoede",
          "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
          "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
          "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
          "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"]


def client(key):
    return genai.Client(api_key=key)


def to_np(seg):
    x = np.array(seg.set_channels(1).get_array_of_samples(), dtype=np.float32)
    return x / float(1 << (8 * seg.sample_width - 1))


def wav_bytes(seg):
    b = io.BytesIO()
    seg.export(b, format="wav")
    return b.getvalue()


def load_manifest(raw):
    data = json.loads(raw)
    segs = data["segments"] if isinstance(data, dict) else data
    for i, s in enumerate(segs):
        s.setdefault("id", i)
    return segs


def find_segments(manifest, a_ms, b_ms):
    hits = [(min(s["end_ms"], b_ms) - max(s["start_ms"], a_ms), s) for s in manifest]
    return [s for ov, s in sorted(hits, key=lambda h: -h[0]) if ov > 0]


def _json(text):
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


def diagnose(c, model, clip, text, dialect, word=None, wi=None, correction=None):
    """Return {'word':..., 'text': rewritten sentence}. The reviewer has flagged `word`
    (or only the sentence, if word is None)."""
    rule = (f"Rewrite the sentence so a TTS engine pronounces it exactly as a native "
            f"{dialect} speaker would say it. Change only the problem word (spelling / "
            f"diacritics); keep every other word identical.")
    if word:
        task = (f'The reviewer heard the word "{word}" (word number {wi + 1} of the sentence) '
                f"pronounced wrongly in this audio. ")
        task += (f"It should be pronounced: {correction}. " if correction else
                 "Decide how to respell it so the TTS says it correctly. ")
    else:
        task = ("A word in this sentence is mispronounced in the audio. Listen, find it, "
                "and fix it. " if not correction else
                f"A word in this sentence is mispronounced. The correct pronunciation is: {correction}. ")
    prompt = (f"Sentence: {text}\n{task}{rule}\n"
              'Return JSON only: {"word": "the word you changed", "text": "full sentence"}')
    part = types.Part.from_bytes(data=wav_bytes(clip), mime_type="audio/wav")
    return _json(c.models.generate_content(model=model, contents=[part, prompt]).text)


def _trim(seg, keep=80):
    s = detect_leading_silence(seg, -45)
    e = detect_leading_silence(seg.reverse(), -45)
    return seg[max(0, s - keep): len(seg) - max(0, e - keep)]


def tts(c, model, voice, text, style, like):
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    r = c.models.generate_content(model=model, config=cfg,
                                  contents=f"{style}\n\n{text}" if style else text)
    pcm = r.candidates[0].content.parts[0].inline_data.data
    seg = AudioSegment(data=pcm, sample_width=2, frame_rate=24000, channels=1)
    seg = _trim(seg).set_frame_rate(like.frame_rate).set_channels(like.channels)
    return seg.set_sample_width(like.sample_width)


def _stats(seg, text):
    x, sr = to_np(seg), seg.frame_rate
    f0 = librosa.pyin(x, fmin=60, fmax=450, sr=sr)[0]
    f0 = f0[~np.isnan(f0)]
    return dict(f0=float(np.median(f0)) if len(f0) else np.nan, db=seg.dBFS,
                cps=len(text.replace(" ", "")) / max(seg.duration_seconds, 0.1))


def compare(new, text, refs):
    """Compare new clip with neighbouring sentences [(seg, text)]."""
    s = _stats(new, text)
    r = [_stats(g, t) for g, t in refs]
    f0r = np.nanmedian([x["f0"] for x in r])
    pitch = abs(12 * np.log2(s["f0"] / f0r)) if not np.isnan(s["f0"] * f0r) else 0.0
    pace = abs(np.log2(s["cps"] / np.median([x["cps"] for x in r])))
    return dict(pitch_semitones=round(pitch, 2), pace_log2=round(pace, 2),
                gain_db=round(float(np.median([x["db"] for x in r]) - s["db"]), 1),
                score=pitch / 2 + pace / 0.25, ok=pitch <= 2 and pace <= 0.25)


def neighbours(audio, manifest, seg):
    i = manifest.index(seg)
    return [(audio[m["start_ms"]:m["end_ms"]], m["text"])
            for m in manifest[max(0, i - 1): i + 2] if m is not seg]


def regenerate(c, model, voice, text, style, audio, manifest, seg, tries=3):
    refs = neighbours(audio, manifest, seg)
    like = audio[seg["start_ms"]:seg["end_ms"]]
    best = None
    for _ in range(tries):
        new = tts(c, model, voice, text, style, like)
        m = compare(new, text, refs) if refs else dict(score=0, ok=True, gain_db=0)
        new = new.apply_gain(m["gain_db"])
        if best is None or m["score"] < best[1]["score"]:
            best = (new, m)
        if m["ok"]:
            break
    return best


def splice(audio, manifest, seg, new, text, fade=20):
    def join(x, y):
        if not len(x): return y
        if not len(y): return x
        return x.append(y, crossfade=min(fade, len(x), len(y)))
    a, b = seg["start_ms"], seg["end_ms"]
    out = join(join(audio[:a], new), audio[b:])
    delta = len(out) - len(audio)
    mf = copy.deepcopy(manifest)
    for m in mf:
        if m["id"] == seg["id"]:
            m["end_ms"] += delta
            m["tts_text"] = text
        elif m["start_ms"] >= b:
            m["start_ms"] += delta
            m["end_ms"] += delta
    return out, mf


def split_script(text):
    parts = re.split(r"(?<=[.!?؟…])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def build_manifest(c, model, audio, sentences, win_s=240, batch=40, progress=None):
    """Align script sentences to audio. Returns (rows, warning_or_None).
    Sentences that cannot be located are skipped, given estimated times and marked
    status='estimated' so the user can fix them by hand."""
    total = len(audio)
    sil = detect_silence(audio, min_silence_len=250, silence_thresh=audio.dBFS - 16)
    mids = [(a + b) // 2 for a, b in sil]

    def snap(ms):
        if not mids:
            return int(ms)
        m = min(mids, key=lambda x: abs(x - ms))
        return int(m if abs(m - ms) <= 1200 else ms)

    def report(done, pos):
        if progress:
            progress(min(done / n, 1.0), f"Aligned {done}/{n} sentences "
                     f"({100 * done // n}%) - audio position {pos / 1000:.0f}s of {total / 1000:.0f}s")

    n = len(sentences)
    found, pos, idx, fails, stopped = {}, 0, 0, 0, False
    report(0, 0)
    while idx < n and pos < total - 500:
        got, start = [], 0
        for attempt in range(2):  # 2nd try: look further back, wider window
            start = max(0, pos - (4000 if attempt == 0 else 10000)) if pos else 0
            clip = audio[start: start + int(win_s * (1 + 0.5 * attempt)) * 1000]
            listing = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences[idx: idx + batch]))
            prompt = ("The audio is a recording of a script. The clip starts at 0 seconds and may "
                      "begin with a few words from an earlier sentence - ignore those. Below are "
                      "the next script sentences in order (the speech may differ slightly in "
                      "dialect, match by meaning).\nFor each sentence that is FULLY spoken in this "
                      "clip, give its start and end time in seconds (decimal). Stop at the first "
                      "sentence that is not fully inside the clip.\n" + listing +
                      '\nReturn JSON only: {"items":[{"i":0,"start":0.0,"end":4.2}]}')
            part = types.Part.from_bytes(data=wav_bytes(clip), mime_type="audio/wav")
            try:
                items = sorted(_json(c.models.generate_content(
                    model=model, contents=[part, prompt]).text)["items"], key=lambda d: d["i"])
            except Exception:
                items = []
            got = []
            for k, it in enumerate(items):
                try:
                    ok = it["i"] == k and float(it["end"]) > float(it["start"])
                except Exception:
                    ok = False
                if not ok:
                    break
                got.append(it)
            if got:
                break
        if not got:  # skip this sentence, keep going (stop after 3 misses in a row)
            fails += 1
            idx += 1
            report(idx, pos)
            if fails >= 3:
                stopped = True
                break
            continue
        fails = 0
        for it in got:
            found[idx + it["i"]] = (start + float(it["start"]) * 1000, start + float(it["end"]) * 1000)
        idx += len(got)
        end_ms = found[idx - 1][1]
        pos = snap(end_ms) if snap(end_ms) > pos else int(end_ms)
        report(idx, pos)

    # boundaries: snapped between aligned neighbours, text-length estimates inside gaps
    b, i = [0], 0
    while i < n:
        if i in found:
            if i + 1 == n:
                b.append(total)
            elif i + 1 in found:
                b.append(snap((found[i][1] + found[i + 1][0]) / 2))
            else:
                b.append(snap(found[i][1]))
            i += 1
        else:
            j = i
            while j < n and j not in found:
                j += 1
            run_end = total if j == n else snap(found[j][0])
            w = [max(len(s), 1) for s in sentences[i:j]]
            acc = 0
            for x in w:
                acc += x
                b.append(int(b[i] + (run_end - b[i]) * acc / sum(w)))
            i = j
    for k in range(1, len(b)):
        b[k] = max(b[k], b[k - 1] + 1)
    rows = [dict(id=k, text=sentences[k], start_ms=int(b[k]), end_ms=int(b[k + 1]),
                 status="ok" if k in found else "estimated") for k in range(n)]
    missing = [k + 1 for k in range(n) if k not in found]
    msg = None
    if missing:
        shown = ", ".join(map(str, missing[:15])) + (" ..." if len(missing) > 15 else "")
        msg = (f"Aligned {len(found)} of {n} sentences. Not located: {shown}"
               + (" (stopped after 3 misses in a row)." if stopped else ".")
               + " Rows marked 'estimated' are rough guesses - fix their start/end by hand.")
    return rows, msg
