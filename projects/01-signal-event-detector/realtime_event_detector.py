"""
Real-time sensor/audio event detector.

Pipeline:
    microphone block
    -> raw ring buffer
    -> high-pass filter
    -> squared signal energy
    -> moving-average energy envelope
    -> robust adaptive threshold
    -> delayed event clip saving with pre/post audio

"""

import json
import os
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import butter, lfilter

# ============================
# CONFIG
# ============================

FS = 44100
CHANNELS = 1
BLOCKSIZE = 1024
BUFFER_SECONDS = 10

ENVELOPE_MS = 10

# Detection sensitivity: trigger when score > noise_mu + THRESH_STD * noise_sigma
THRESH_STD = 6.0

# Hysteresis: event ends only after score drops below this lower threshold.
HYST_RATIO = 0.6

# Minimum time between event starts.
REFRACTORY_SEC = 0.75

# Learn baseline noise before detection starts.
CALIBRATION_SEC = 3.0

# Slope gating is off by default. The basic detector should be understood first.
USE_SLOPE_GATE = False
SLOPE_STD = 4.0

CLIP_PRE_SEC = 0.5
CLIP_POST_SEC = 1.5

# Noise model adaptation speed.
NOISE_TAU_SEC = 2.0

# High-pass filter.
HP_CUTOFF = 150
HP_ORDER = 4

# Percentile score is more robust than max because it ignores isolated spikes.
SCORE_PERCENTILE = 95

OUTPUT_DIR = Path("outputs/audio/events")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PATH = OUTPUT_DIR / "events.jsonl"

# ============================
# RING BUFFER
# ============================

BUFFER_LEN = int(FS * BUFFER_SECONDS)
ring = np.zeros(BUFFER_LEN, dtype=np.float32)
write_idx = 0
sample_counter = 0  # Absolute number of samples written since program start.

# ============================
# ENVELOPE FILTER: MOVING AVERAGE FIR
# ============================

# M is the number of samples in the moving-average window.
# If ENVELOPE_MS = 10 and FS = 44100, M = 441 samples.
M = max(int(FS * ENVELOPE_MS / 1000), 1)
h = np.ones(M, dtype=np.float32) / M
zi_env = np.zeros(M - 1, dtype=np.float32)

# ============================
# HIGHPASS FILTER
# ============================

b_hp, a_hp = butter(HP_ORDER, HP_CUTOFF, btype="highpass", fs=FS)
zi_hp = np.zeros(max(len(a_hp), len(b_hp)) - 1, dtype=np.float32)

# ============================
# NOISE MODEL
# ============================

dt_block = BLOCKSIZE / FS
alpha = dt_block / NOISE_TAU_SEC
alpha = min(max(alpha, 0.001), 0.05)

noise_mu = 0.0
noise_sigma = 1e-9

# ============================
# DETECTOR STATE
# ============================

last_event_sample = -10**18
event_counter = 0
block_counter = 0

armed = False
in_event = False
pending_event = None  # Holds event info until enough post-trigger audio exists.

audio_queue = queue.Queue(maxsize=50)


# ============================
# HELPER FUNCTIONS
# ============================

def audio_callback(indata, frames, time_info, status):
    """Audio callback should stay tiny: copy input and push it into a queue."""
    if status:
        print(status)

    block = indata[:, 0].astype(np.float32, copy=True)

    try:
        audio_queue.put_nowait(block)
    except queue.Full:
        # Dropping is better than blocking inside the real-time callback.
        pass


def write_ring_buffer(block):
    """Write one block into the circular raw-audio ring buffer."""
    global write_idx

    n = len(block)
    end_idx = write_idx + n

    if end_idx <= BUFFER_LEN:
        ring[write_idx:end_idx] = block
    else:
        part1 = BUFFER_LEN - write_idx
        ring[write_idx:] = block[:part1]
        ring[: end_idx % BUFFER_LEN] = block[part1:]

    write_idx = end_idx % BUFFER_LEN


def read_ring_buffer_window(start_sample, end_sample):
    """
    Read [start_sample, end_sample) from the raw-audio ring buffer.

    start_sample and end_sample are absolute sample indices, not ring indices.
    This works as long as the requested window has not been overwritten.
    """
    length = int(end_sample - start_sample)
    if length <= 0:
        return np.array([], dtype=np.float32)
    if length > BUFFER_LEN:
        raise ValueError("Requested clip is longer than the ring buffer.")

    start_idx = int(start_sample % BUFFER_LEN)
    end_idx = start_idx + length

    if end_idx <= BUFFER_LEN:
        return ring[start_idx:end_idx].copy()

    return np.concatenate((ring[start_idx:], ring[: end_idx % BUFFER_LEN])).copy()


def compute_envelope(block):
    """
    Convert raw audio into a smooth short-term energy envelope.

    Math:
        1. High-pass filter removes slow/low-frequency components.
        2. Squaring converts signed waveform x[n] into energy-like x[n]^2.
        3. Moving average smooths energy over ENVELOPE_MS milliseconds.
    """
    global zi_hp, zi_env

    block_hp, zi_hp[:] = lfilter(b_hp, a_hp, block, zi=zi_hp)
    energy = block_hp * block_hp
    envelope_block, zi_env[:] = lfilter(h, [1.0], energy, zi=zi_env)
    return envelope_block.astype(np.float32, copy=False)


def robust_stats(x):
    """
    Robust baseline statistics using median and MAD.

    MAD = median(|x - median(x)|)
    For normal-like data, sigma ≈ 1.4826 * MAD.
    """
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = float(1.4826 * mad)
    return med, max(sigma, 1e-12)


def update_noise_model(mu, sigma, block_median, block_sigma):
    """Exponential moving average update for the noise baseline."""
    new_mu = (1.0 - alpha) * mu + alpha * block_median
    new_sigma = (1.0 - alpha) * sigma + alpha * block_sigma
    return new_mu, max(new_sigma, 1e-12)


def compute_slope_gate(envelope_block):
    """
    Optional slope gate with matching units.

    The original prototype compared diff(envelope) against noise_sigma from the
    envelope level. That mixes units. Here slope is compared against robust
    statistics of slope itself.
    """
    slope = np.diff(envelope_block)
    if len(slope) == 0:
        return True, 0.0, 0.0

    slope_peak = float(np.max(slope))
    slope_med, slope_sigma = robust_stats(slope)
    slope_threshold = slope_med + SLOPE_STD * slope_sigma
    return slope_peak > slope_threshold, slope_peak, slope_threshold


def save_event_clip(event, current_sample):
    """Save raw event audio and append metadata as one JSON line."""
    pre_samples = int(FS * CLIP_PRE_SEC)
    post_samples = int(FS * CLIP_POST_SEC)

    trigger_sample = int(event["trigger_sample"])
    start_sample = max(0, trigger_sample - pre_samples)
    end_sample = trigger_sample + post_samples

    if current_sample < end_sample:
        # Not enough future audio exists yet.
        return False

    if current_sample - start_sample > BUFFER_LEN:
        print("Warning: event clip was overwritten before it could be saved.")
        return True

    clip = read_ring_buffer_window(start_sample, end_sample)

    # Remove DC offset, but do NOT normalize. Preserving amplitude helps analysis.
    clip = clip - float(np.mean(clip))
    clip = np.clip(clip, -1.0, 1.0)
    clip_int16 = (clip * 32767).astype(np.int16)

    filename = OUTPUT_DIR / f"event_{event['event_id']:04d}.wav"
    wavfile.write(filename, FS, clip_int16)

    record = {
        "event_id": event["event_id"],
        "filename": str(filename),
        "trigger_sample": trigger_sample,
        "trigger_time_sec": trigger_sample / FS,
        "clip_start_sample": start_sample,
        "clip_end_sample": end_sample,
        "clip_start_time_sec": start_sample / FS,
        "clip_end_time_sec": end_sample / FS,
        "score": event["score"],
        "peak": event["peak"],
        "threshold_high": event["threshold_high"],
        "threshold_low": event["threshold_low"],
        "noise_mu": event["noise_mu"],
        "noise_sigma": event["noise_sigma"],
        "slope_peak": event["slope_peak"],
        "slope_threshold": event["slope_threshold"],
        "score_percentile": SCORE_PERCENTILE,
    }

    with METADATA_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"Saved {filename}")
    return True


print("Starting real-time detector. Press Ctrl+C to stop.")
print(f"Writing events to: {OUTPUT_DIR}")
print(f"Writing metadata to: {METADATA_PATH}")
start_time = time.time()

try:
    with sd.InputStream(
        samplerate=FS,
        channels=CHANNELS,
        blocksize=BLOCKSIZE,
        callback=audio_callback,
    ):
        while True:
            block = audio_queue.get()
            block_counter += 1

            t0 = time.perf_counter()
            now = time.time()

            # ----------------------------
            # Write raw audio to ring buffer
            # ----------------------------
            n = len(block)
            block_start_sample = sample_counter
            write_ring_buffer(block)
            sample_counter += n
            current_sample = sample_counter

            # ----------------------------
            # Compute envelope used for detection
            # ----------------------------
            envelope_block = compute_envelope(block)

            # Peak is useful metadata, but score is better for detection.
            peak_idx = int(np.argmax(envelope_block))
            peak = float(envelope_block[peak_idx])
            score = float(np.percentile(envelope_block, SCORE_PERCENTILE))

            # Robust block statistics for baseline adaptation.
            med, sigma_est = robust_stats(envelope_block)

            # ----------------------------
            # Calibration phase
            # ----------------------------
            if not armed:
                noise_mu, noise_sigma = update_noise_model(
                    noise_mu, noise_sigma, med, sigma_est
                )

                if (now - start_time) >= CALIBRATION_SEC:
                    armed = True
                    print(f"Detector armed after {CALIBRATION_SEC}s calibration.")
                    print(
                        f"Initial baseline mu={noise_mu:.3e}, "
                        f"sigma={noise_sigma:.3e}"
                    )

                continue

            # ----------------------------
            # Thresholds
            # ----------------------------
            threshold_high = noise_mu + THRESH_STD * noise_sigma
            threshold_low = noise_mu + HYST_RATIO * THRESH_STD * noise_sigma

            # Optional slope gate.
            slope_ok, slope_peak, slope_threshold = compute_slope_gate(envelope_block)
            if not USE_SLOPE_GATE:
                slope_ok = True

            # ----------------------------
            # Save pending clip after post-event audio exists
            # ----------------------------
            if pending_event is not None:
                saved_or_done = save_event_clip(pending_event, current_sample)
                if saved_or_done:
                    pending_event = None

            # ----------------------------
            # State machine detection
            # ----------------------------
            refractory_samples = int(FS * REFRACTORY_SEC)

            if not in_event:
                can_trigger = (current_sample - last_event_sample) > refractory_samples

                if score > threshold_high and slope_ok and can_trigger:
                    event_counter += 1
                    in_event = True

                    # Approximate trigger location as the peak envelope sample in this block.
                    trigger_sample = block_start_sample + peak_idx
                    last_event_sample = trigger_sample

                    pending_event = {
                        "event_id": event_counter,
                        "trigger_sample": trigger_sample,
                        "score": score,
                        "peak": peak,
                        "threshold_high": threshold_high,
                        "threshold_low": threshold_low,
                        "noise_mu": noise_mu,
                        "noise_sigma": noise_sigma,
                        "slope_peak": slope_peak,
                        "slope_threshold": slope_threshold,
                    }

                    print(
                        f"[EVENT {event_counter}] "
                        f"score={score:.3e} peak={peak:.3e} "
                        f"thr={threshold_high:.3e} "
                        f"trigger_t={trigger_sample / FS:.3f}s"
                    )
                else:
                    # Update baseline only when not triggered and not inside an event.
                    noise_mu, noise_sigma = update_noise_model(
                        noise_mu, noise_sigma, med, sigma_est
                    )
            else:
                # Event stays active until the score drops below lower threshold.
                if score < threshold_low:
                    in_event = False

            # ----------------------------
            # Performance metrics
            # ----------------------------
            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000
            budget_ms = (BLOCKSIZE / FS) * 1000
            util = dt_ms / budget_ms

            if block_counter % 50 == 0:
                print(
                    f"proc={dt_ms:.3f}ms util={util:.3f} "
                    f"mu={noise_mu:.3e} sig={noise_sigma:.3e} "
                    f"score={score:.3e} peak={peak:.3e} "
                    f"thrH={threshold_high:.3e} thrL={threshold_low:.3e}"
                )

except KeyboardInterrupt:
    print("\nStopped.")
