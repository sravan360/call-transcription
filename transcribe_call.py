"""
Transcribe a phone-call recording into an Agent / Customer transcript with timestamps.

Stereo recordings (each party on its own channel):
    Each channel is transcribed separately, so speaker labels are exact.
    Stereo files whose channels carry the same mix (or where one channel
    is silent) are detected and handled like mono.

Mono recordings (both voices mixed together):
    Whisper transcribes with word timings. A voice fingerprint (SpeechBrain
    ECAPA model) is taken every 0.5 s over sliding 1.5 s windows, the
    windows are clustered into speakers, and every word gets the speaker
    of the window around it - so quick back-and-forth without pauses is
    still split correctly. The Customer is the speaker who uses the fewest
    agent-style phrases ("thank you for calling", "my name is" ...); every
    other speaker is labelled Agent (handy for transferred calls).

Setup (once, inside your venv):
    pip install faster-whisper speechbrain scikit-learn
    CPU:  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    GPU:  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

Usage:
    python transcribe_call.py call.mp3
    python transcribe_call.py call.mp3 --speakers 3          # call was transferred between agents
    python transcribe_call.py call.mp3 --agent-channel right # stereo, labels swapped
    python transcribe_call.py call.mp3 --swap-speakers       # mono, labels swapped
    python transcribe_call.py call.mp3 --force-mono          # ignore stereo channels
    python transcribe_call.py call.mp3 --model large-v3 -o out.json
    python transcribe_call.py call.mp3 --format txt          # readable [mm:ss - mm:ss] lines
    python transcribe_call.py call.mp3 --batch-size 16 --compute-type int8_float16  # faster on GPU

Output (default JSON):
    [{"text": "...", "start": 0.45, "end": 1.21, "user_role_classified": "customer"}, ...]

GPU (NVIDIA, CUDA 12): used automatically when available, otherwise CPU.
    Force one with --device cuda / --device cpu.

Runs fully offline after the models are downloaded on first use.
"""

import argparse
import json
import re
from pathlib import Path

import av
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel

TARGET_RATE = 16000  # Whisper and ECAPA both expect 16 kHz mono float32
SCRIPT_DIR = Path(__file__).resolve().parent

WINDOW = 1.5  # seconds of audio per voice fingerprint
HOP = 0.5     # seconds between fingerprints

AGENT_PHRASES = [
    r"thank you for calling", r"thanks for calling", r"how (can|may) i (help|assist)",
    r"my name is", r"this is \w+ (from|with|calling)", r"calling from", r"customer (care|service)",
    r"is there anything else", r"for (security|verification)", r"verify",
    r"have a (great|nice|good) day", r"thank you for (your patience|holding|waiting)",
    r"please hold", r"bear with me", r"give me (a|one) (quick )?(second|moment|minute)",
    r"i('ll| will) (check|look|transfer|connect|send)", r"let me (check|connect|transfer)",
    r"can you (please )?(provide|confirm|spell)", r"you('re| are) (very )?welcome",
    r"account number", r"date of birth", r"recorded", r"quality (and|&) training",
]


# ---------------------------------------------------------------- audio

def load_channels(path):
    """Decode the file and return a list of per-channel 16 kHz float32 arrays."""
    container = av.open(str(path))
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=TARGET_RATE)
    chunks = []
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray())
    for out in resampler.resample(None):  # flush
        chunks.append(out.to_ndarray())
    audio = np.concatenate(chunks, axis=1).astype(np.float32)
    return [audio[ch] for ch in range(audio.shape[0])]


def stereo_is_separated(left, right):
    """True only if the two channels really carry different speakers."""
    rms_l, rms_r = np.sqrt(np.mean(left ** 2)), np.sqrt(np.mean(right ** 2))
    corr = float(np.corrcoef(left, right)[0, 1]) if rms_l > 0 and rms_r > 0 else 1.0
    ratio = min(rms_l, rms_r) / (max(rms_l, rms_r) + 1e-12)
    print(f"Stereo check: channel correlation {corr:.2f}, quieter/louder level {ratio:.2f}")
    if corr > 0.6:
        print("  -> both channels carry the same mix; using voice-based separation")
        return False
    if ratio < 0.05:
        print("  -> one channel is (almost) silent; using voice-based separation")
        return False
    print("  -> channels are separate speakers; using channel-based separation")
    return True


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def pick_device(requested):
    if requested != "auto":
        return requested
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------- stereo

def transcribe(model, audio, speaker, language):
    segments, _ = model.transcribe(
        audio,
        language=language,
        vad_filter=True,  # skip silence while the other party talks
        beam_size=5,
        condition_on_previous_text=False,  # reduces repeated/hallucinated lines
    )
    return [
        {"speaker": speaker, "start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
        if s.text.strip()
    ]


# ---------------------------------------------------------------- mono

def transcribe_words(model, audio, language):
    segments, _ = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    return [
        {"start": w.start, "end": w.end, "text": w.word}
        for s in segments for w in (s.words or []) if w.word.strip()
    ]


def load_speaker_model(device):
    from speechbrain.inference.speaker import EncoderClassifier

    kwargs = {
        "source": "speechbrain/spkrec-ecapa-voxceleb",
        "savedir": str(SCRIPT_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"),
        "run_opts": {"device": device},
    }
    try:  # Windows can't create symlinks without admin rights - copy files instead
        from speechbrain.utils.fetching import LocalStrategy
        kwargs["local_strategy"] = LocalStrategy.COPY
    except ImportError:
        pass
    return EncoderClassifier.from_hparams(**kwargs)


def speech_windows(words, duration):
    """Start times of sliding windows that are mostly speech."""
    mask = np.zeros(int(duration / 0.05) + 1, dtype=bool)  # 50 ms resolution
    for w in words:
        mask[int(w["start"] / 0.05): int(w["end"] / 0.05) + 1] = True
    starts = []
    t = 0.0
    while t + WINDOW <= duration + 1e-6:
        if mask[int(t / 0.05): int((t + WINDOW) / 0.05)].mean() >= 0.5:
            starts.append(t)
        t += HOP
    if not starts and words:  # very short recording
        starts = [max(0.0, min(words[0]["start"], duration - WINDOW))]
    return np.array(starts)


def embed_windows(encoder, audio, starts, device, batch=64):
    import torch

    n = int(WINDOW * TARGET_RATE)
    padded = np.pad(audio, (0, n))
    clips = np.stack([padded[int(s * TARGET_RATE): int(s * TARGET_RATE) + n] for s in starts])
    embs = []
    with torch.no_grad():
        for i in range(0, len(clips), batch):
            x = torch.from_numpy(clips[i:i + batch]).to(device)
            embs.append(encoder.encode_batch(x).squeeze(1).cpu().numpy())
    embs = np.concatenate(embs)
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)


def cluster_windows(embs, starts, n_speakers):
    from sklearn.cluster import AgglomerativeClustering

    if len(embs) <= n_speakers:
        return np.zeros(len(embs), dtype=int)
    # Ward linkage on unit vectors gives balanced voice groups instead of isolating outliers
    labels = AgglomerativeClustering(n_clusters=n_speakers, linkage="ward").fit_predict(embs)
    # Majority filter over neighbouring windows to remove single-window flips
    smoothed = labels.copy()
    for i in range(1, len(labels) - 1):
        if starts[i + 1] - starts[i - 1] <= 2 * HOP + 1e-6 and labels[i - 1] == labels[i + 1]:
            smoothed[i] = labels[i - 1]
    return smoothed


def label_words(words, starts, labels):
    centres = starts + WINDOW / 2
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        w["spk"] = int(labels[np.abs(centres - mid).argmin()])


def name_speakers(words, n_speakers, swap):
    """Customer = speaker with fewest agent phrases (ties: not the first speaker). Others = Agent."""
    texts = {k: [] for k in range(n_speakers)}
    for w in words:
        texts[w["spk"]].append(w["text"])
    share = {k: len(v) / max(1, len(words)) for k, v in texts.items()}
    score = {k: sum(len(re.findall(p, "".join(v).lower())) for p in AGENT_PHRASES) for k, v in texts.items()}
    first = words[0]["spk"]
    candidates = [k for k in texts if share[k] >= 0.1] or list(texts)
    order = sorted(candidates, key=lambda k: (score[k], k == first, -share[k]))
    customer = order[1] if swap and len(order) > 1 else order[0]
    for k in texts:
        role = "Customer" if k == customer else "Agent"
        print(f"  voice {k}: {share[k]:.0%} of words, {score[k]} agent phrases -> {role}")
    return {k: ("Customer" if k == customer else "Agent") for k in texts}


def diarize_mono(model, audio, language, swap, device, n_speakers):
    print("Transcribing (word timings) ...")
    words = transcribe_words(model, audio, language)
    if not words:
        return []
    starts = speech_windows(words, len(audio) / TARGET_RATE)
    print(f"Separating {n_speakers} voices using {len(starts)} fingerprint windows ...")
    encoder = load_speaker_model(device)
    labels = cluster_windows(embed_windows(encoder, audio, starts, device), starts, n_speakers)
    label_words(words, starts, labels)
    names = name_speakers(words, n_speakers, swap)
    # Keep Whisper's own spacing (" Hello", "-huh", ".com") so words re-join cleanly
    return [
        {"speaker": names[w["spk"]], "start": w["start"], "end": w["end"], "text": w["text"], "word": True}
        for w in words
    ]


# ---------------------------------------------------------------- output

def merge_turns(segments, gap=1.5):
    """Join consecutive segments from the same speaker into one turn."""
    turns = []
    for seg in segments:
        prev = turns[-1] if turns else None
        if prev and prev["speaker"] == seg["speaker"] and seg["start"] - prev["end"] <= gap:
            prev["text"] += seg["text"] if seg.get("word") else " " + seg["text"]
            prev["end"] = seg["end"]
        else:
            turns.append(dict(seg))
    for t in turns:
        t["text"] = t["text"].strip()
    return turns


class BatchedModel:
    """Runs VAD speech chunks through Whisper in parallel batches; same transcribe() call."""

    def __init__(self, model, batch_size):
        self.pipeline = BatchedInferencePipeline(model)
        self.batch_size = batch_size

    def transcribe(self, audio, **kwargs):
        return self.pipeline.transcribe(audio, batch_size=self.batch_size, **kwargs)


def load_model(name="medium.en", device="auto", compute_type=None, batch_size=0):
    """Return (model, resolved device). batch_size > 0 enables batched inference."""
    device = pick_device(device)
    compute_type = compute_type or ("float16" if device == "cuda" else "int8")
    print(f"Loading Whisper model '{name}' on {device} ({compute_type}"
          f"{f', batch {batch_size}' if batch_size > 0 else ''}) (first run downloads it) ...")
    model = WhisperModel(name, device=device, compute_type=compute_type)
    return (BatchedModel(model, batch_size) if batch_size > 0 else model), device


def transcribe_call(audio_path, model, device, language="en", agent_channel="left",
                    speakers=2, swap_speakers=False, force_mono=False):
    """Transcribe one recording and return merged speaker turns."""
    print(f"Loading {audio_path} ...")
    channels = load_channels(audio_path)
    print(f"{len(channels)} channel(s), {len(channels[0]) / TARGET_RATE:.0f} s")

    use_channels = (
        len(channels) >= 2 and not force_mono and stereo_is_separated(channels[0], channels[1])
    )
    if use_channels:
        agent_idx = 0 if agent_channel == "left" else 1
        labels = {agent_idx: "Agent", 1 - agent_idx: "Customer"}
        segments = []
        for idx in (0, 1):
            print(f"Transcribing {labels[idx]} channel ...")
            segments += transcribe(model, channels[idx], labels[idx], language)
    else:
        mono = np.mean(channels, axis=0).astype(np.float32)
        segments = diarize_mono(model, mono, language, swap_speakers, device, speakers)

    segments.sort(key=lambda s: s["start"])
    return merge_turns(segments)


def turns_to_records(turns):
    """JSON output format: [{text, start, end, user_role_classified}, ...]"""
    return [
        {
            "text": t["text"],
            "start": float(t["start"]),
            "end": float(t["end"]),
            "user_role_classified": t["speaker"].lower(),
        }
        for t in turns
    ]


def main():
    p = argparse.ArgumentParser(description="Agent / Customer transcript of a phone call.")
    p.add_argument("audio", help="Path to the recording (mp3, wav, m4a, ...)")
    p.add_argument("-o", "--output", help="Output path (default: <audio>_transcript.json / .txt)")
    p.add_argument("--format", choices=["json", "txt"], default="json",
                   help="json: list of {text, start, end, user_role_classified}; txt: readable lines (default: json)")
    p.add_argument("--agent-channel", choices=["left", "right"], default="left",
                   help="Stereo: which channel is the Agent (default: left)")
    p.add_argument("--speakers", type=int, default=2,
                   help="Mono: number of distinct voices, e.g. 3 if the call was transferred (default: 2)")
    p.add_argument("--swap-speakers", action="store_true",
                   help="Mono: pick a different voice as the Customer if the automatic choice is wrong")
    p.add_argument("--force-mono", action="store_true",
                   help="Ignore stereo channels and use voice-based speaker separation")
    p.add_argument("--model", default="medium.en",
                   help="Whisper model: tiny.en, base.en, small.en, medium.en, large-v3 (default: medium.en)")
    p.add_argument("--language", default="en")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Run on GPU (cuda) or CPU (default: auto-detect)")
    p.add_argument("--compute-type",
                   help="e.g. float16, int8_float16, int8 (default: float16 on GPU, int8 on CPU)")
    p.add_argument("--batch-size", type=int, default=0,
                   help="Batched inference: transcribe this many speech chunks at once (default: 0 = off)")
    args = p.parse_args()

    audio_path = Path(args.audio)
    out_path = (
        Path(args.output) if args.output
        else audio_path.with_name(f"{audio_path.stem}_transcript.{args.format}")
    )

    model, device = load_model(args.model, args.device, args.compute_type, args.batch_size)
    turns = transcribe_call(
        audio_path, model, device, args.language, args.agent_channel,
        args.speakers, args.swap_speakers, args.force_mono,
    )

    if args.format == "json":
        output = json.dumps(turns_to_records(turns), indent=2, ensure_ascii=False)
    else:
        lines = [f"Transcript: {audio_path.name}", ""]
        for t in turns:
            lines.append(f"[{fmt_time(t['start'])} - {fmt_time(t['end'])}] {t['speaker']}: {t['text']}")
        output = "\n".join(lines)
    out_path.write_text(output + "\n", encoding="utf-8")

    print(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
