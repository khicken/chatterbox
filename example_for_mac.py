import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from datetime import datetime
import numpy as np
import logging
import pyrubberband as pyrb

logger = logging.getLogger("Test")

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info("Using device: " + device)
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

def reduce_silence(wav, sample_rate, silence_threshold=0.01, max_silence_duration=0.2):
    """
    Reduce silence in audio to a maximum duration.

    Args:
        wav: Audio tensor (channels, samples)
        sample_rate: Sample rate of the audio
        silence_threshold: Amplitude threshold below which audio is considered silence
        max_silence_duration: Maximum duration of silence in seconds (default 0.2s)

    Returns:
        Audio tensor with reduced silence
    """
    # Convert to numpy for easier processing
    audio = wav.cpu().numpy()

    # Get absolute amplitude (average across channels if stereo)
    if audio.ndim > 1:
        amplitude = np.abs(audio).mean(axis=0)
    else:
        amplitude = np.abs(audio)

    # Detect non-silent regions
    is_silent = amplitude < silence_threshold

    # Find silence and non-silence segments
    max_silence_samples = int(max_silence_duration * sample_rate)

    # Build new audio by keeping all non-silent parts and limiting silent parts
    result_parts = []
    i = 0

    while i < len(is_silent):
        if not is_silent[i]:
            # Non-silent region - find its extent
            start = i
            while i < len(is_silent) and not is_silent[i]:
                i += 1
            # Keep the entire non-silent region
            if audio.ndim > 1:
                result_parts.append(audio[:, start:i])
            else:
                result_parts.append(audio[start:i])
        else:
            # Silent region - find its extent
            start = i
            while i < len(is_silent) and is_silent[i]:
                i += 1
            silence_length = i - start

            # Keep only up to max_silence_samples
            kept_silence = min(silence_length, max_silence_samples)
            if kept_silence > 0:
                if audio.ndim > 1:
                    result_parts.append(audio[:, start:start + kept_silence])
                else:
                    result_parts.append(audio[start:start + kept_silence])

    # Concatenate all parts
    if audio.ndim > 1:
        result = np.concatenate(result_parts, axis=1)
    else:
        result = np.concatenate(result_parts)

    # Convert back to tensor
    return torch.from_numpy(result).to(wav.device)

model = ChatterboxTTS.from_pretrained(device=device)
text = ""

AUDIO_PROMPT_PATH = "morgan_freeman.wav"


for ind, chunk in enumerate([c.strip() for c in text.split("\n") if c.strip()]):
    wav = model.generate(
        chunk,
        audio_prompt_path=AUDIO_PROMPT_PATH,
        exaggeration=0.8,
        cfg_weight=0.5
    )

    # # Reduce silence to max 0.2 seconds
    # wav = reduce_silence(wav, model.sr, silence_threshold=0.01, max_silence_duration=0.2)

    # # Speed up audio by 1.1x without pitch warping using time-stretching
    # wav_np = wav.cpu().numpy()
    # wav_stretched = pyrb.time_stretch(wav_np, model.sr, 1.1)
    # wav = torch.from_numpy(wav_stretched).to(wav.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ta.save(f"output/{timestamp}_audio_chunk_{ind}.wav", wav, model.sr)
