"""
Offline event detector for saved WAV files.

Pipeline:
    .wav file
    -> convert to mono float32 in [-1, 1]
    -> process in blocks
    -> high-pass filter
    -> squared energy
    -> moving-average envelope
    -> robust adaptive threshold
    -> delayed pre/post event extraction
    -> event clips + metadata + optional debug plots

Example:
    python analyze_saved_audio_events.py --input outputs/audio --output outputs/audio/offline_events --plots
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter


@dataclass
class DetectorConfig:
    blocksize: int = 1024
    envelope_ms: float = 10.0
    thresh_std: float = 6.0
    hyst_ratio: float = 0.6
    refractory_sec: float = 0.75
    calibration_sec: float = 3.0
    clip_pre_sec: float = 0.5
    clip_post_sec: float = 1.5
    noise_tau_sec: float = 2.0
    hp_cutoff_hz: float = 150.0
    hp_order: int = 4
    score_percentile: float = 95.0
    min_sigma: float = 1e-12


@dataclass
class PendingEvent:
    event_id: int
    source_file: str
    trigger_sample: int
    score: float
    peak: float
    threshold_high: float
    threshold_low: float
    noise_mu: float
    noise_sigma: float


def iter_wav_files(path: Path) -> Iterable[Path]:
    """Yield one .wav file or all .wav files inside a directory."""
    if path.is_file():
        if path.suffix.lower() != ".wav":
            raise ValueError(f"Input file must be a .wav file: {path}")
        yield path
        return

    if not path.exists():
        raise FileNotFoundError(path)

    yield from sorted(path.rglob("*.wav"))


def wav_to_float32(data: np.ndarray) -> np.ndarray:
    """Convert common WAV sample formats to mono float32 in approximately [-1, 1]."""
    if data.ndim == 2:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float32, copy=False)
    elif data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        # 8-bit PCM is usually unsigned with silence around 128.
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
        max_abs = float(np.max(np.abs(audio))) if len(audio) else 1.0
        if max_abs > 0:
            audio = audio / max_abs

    # Remove NaN/inf just in case a malformed file produces them.
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def robust_stats(x: np.ndarray, min_sigma: float = 1e-12) -> tuple[float, float]:
    """
    Robust median/MAD statistics.

    median gives the baseline level.
    MAD = median(|x - median(x)|) estimates spread while resisting outliers.
    For normal-like noise, sigma ~= 1.4826 * MAD.
    """
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = float(1.4826 * mad)
    return med, max(sigma, min_sigma)


def update_noise_model(
    noise_mu: float,
    noise_sigma: float,
    block_median: float,
    block_sigma: float,
    alpha: float,
    min_sigma: float,
) -> tuple[float, float]:
    """Exponential moving average update for the noise baseline."""
    noise_mu = (1.0 - alpha) * noise_mu + alpha * block_median
    noise_sigma = (1.0 - alpha) * noise_sigma + alpha * block_sigma
    return noise_mu, max(noise_sigma, min_sigma)


def make_envelope_filter(fs: int, cfg: DetectorConfig) -> np.ndarray:
    """Moving-average FIR kernel for smoothing squared signal energy."""
    m = max(int(fs * cfg.envelope_ms / 1000.0), 1)
    return np.ones(m, dtype=np.float32) / m


def compute_block_envelope(
    block: np.ndarray,
    b_hp: np.ndarray,
    a_hp: np.ndarray,
    zi_hp: np.ndarray,
    h_env: np.ndarray,
    zi_env: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a raw block into a short-term energy envelope.

    1. high-pass filter: remove low-frequency rumble/slow drift
    2. square: convert signed waveform into nonnegative energy
    3. moving average: smooth energy over a short window
    """
    block_hp, zi_hp = lfilter(b_hp, a_hp, block, zi=zi_hp)
    energy = block_hp * block_hp
    envelope, zi_env = lfilter(h_env, [1.0], energy, zi=zi_env)
    return envelope.astype(np.float32, copy=False), zi_hp, zi_env


def save_event(
    event: PendingEvent,
    audio: np.ndarray,
    fs: int,
    cfg: DetectorConfig,
    output_dir: Path,
    metadata_file,
    event_global_id: int,
) -> dict:
    """Save a pre/post trigger clip and append one JSON metadata record."""
    pre = int(fs * cfg.clip_pre_sec)
    post = int(fs * cfg.clip_post_sec)

    start_sample = max(0, event.trigger_sample - pre)
    end_sample = min(len(audio), event.trigger_sample + post)
    clip = audio[start_sample:end_sample].copy()

    # Remove DC offset but do not normalize; amplitude is useful information.
    if len(clip):
        clip -= float(np.mean(clip))
    clip = np.clip(clip, -1.0, 1.0)
    clip_int16 = (clip * 32767.0).astype(np.int16)

    source_stem = Path(event.source_file).stem
    clip_name = f"{source_stem}_event_{event_global_id:04d}.wav"
    clip_path = output_dir / clip_name
    wavfile.write(clip_path, fs, clip_int16)

    record = {
        **asdict(event),
        "event_global_id": event_global_id,
        "clip_filename": str(clip_path),
        "trigger_time_sec": event.trigger_sample / fs,
        "clip_start_sample": start_sample,
        "clip_end_sample": end_sample,
        "clip_start_time_sec": start_sample / fs,
        "clip_end_time_sec": end_sample / fs,
        "config": asdict(cfg),
    }
    metadata_file.write(json.dumps(record) + "\n")
    metadata_file.flush()
    return record


def save_debug_plot(
    audio: np.ndarray,
    envelope_trace: np.ndarray,
    thresholds_high: np.ndarray,
    thresholds_low: np.ndarray,
    fs: int,
    records: list[dict],
    wav_path: Path,
    plot_dir: Path,
) -> None:
    """Save simple waveform/envelope plot for offline debugging."""
    import matplotlib.pyplot as plt

    duration = len(audio) / fs
    t_audio = np.arange(len(audio)) / fs
    t_env = np.linspace(0.0, duration, len(envelope_trace), endpoint=False)

    fig = plt.figure(figsize=(12, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t_audio, audio, linewidth=0.6)
    ax1.set_title(f"Waveform: {wav_path.name}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    for rec in records:
        ax1.axvline(rec["trigger_time_sec"], linestyle="--", linewidth=1.0)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t_env, envelope_trace, linewidth=0.8, label="Envelope")
    if len(thresholds_high) == len(envelope_trace):
        ax2.plot(t_env, thresholds_high, linewidth=0.8, label="High threshold")
        ax2.plot(t_env, thresholds_low, linewidth=0.8, label="Low threshold")
    ax2.set_title("Energy envelope and adaptive thresholds")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Energy")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plot_path = plot_dir / f"{wav_path.stem}_debug.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def analyze_file(
    wav_path: Path,
    output_dir: Path,
    metadata_file,
    cfg: DetectorConfig,
    starting_event_id: int,
    make_plots: bool,
) -> tuple[int, list[dict]]:
    """Analyze one WAV file and return the next available global event id."""
    fs, raw = wavfile.read(wav_path)
    audio = wav_to_float32(raw)

    if len(audio) == 0:
        print(f"Skipping empty file: {wav_path}")
        return starting_event_id, []

    # Per-file filters and noise model reset. This keeps files independent.
    b_hp, a_hp = butter(cfg.hp_order, cfg.hp_cutoff_hz, btype="highpass", fs=fs)
    zi_hp = np.zeros(max(len(a_hp), len(b_hp)) - 1, dtype=np.float32)

    h_env = make_envelope_filter(fs, cfg)
    zi_env = np.zeros(len(h_env) - 1, dtype=np.float32)

    dt_block = cfg.blocksize / fs
    alpha = dt_block / cfg.noise_tau_sec
    alpha = min(max(alpha, 0.001), 0.05)

    noise_mu = 0.0
    noise_sigma = cfg.min_sigma
    calibration_samples = int(cfg.calibration_sec * fs)
    refractory_samples = int(cfg.refractory_sec * fs)
    post_samples = int(cfg.clip_post_sec * fs)

    armed = False
    in_event = False
    pending_event: PendingEvent | None = None
    last_event_sample = -10**18
    event_id = starting_event_id
    records_for_file: list[dict] = []

    # For optional plots.
    envelope_trace_parts: list[np.ndarray] = []
    threshold_high_parts: list[np.ndarray] = []
    threshold_low_parts: list[np.ndarray] = []

    for block_start in range(0, len(audio), cfg.blocksize):
        block = audio[block_start : block_start + cfg.blocksize]
        block_end = block_start + len(block)

        envelope, zi_hp, zi_env = compute_block_envelope(
            block, b_hp, a_hp, zi_hp, h_env, zi_env
        )
        envelope_trace_parts.append(envelope.copy())

        peak_idx = int(np.argmax(envelope))
        peak = float(envelope[peak_idx])
        score = float(np.percentile(envelope, cfg.score_percentile))
        med, sigma_est = robust_stats(envelope, cfg.min_sigma)

        if not armed:
            noise_mu, noise_sigma = update_noise_model(
                noise_mu, noise_sigma, med, sigma_est, alpha, cfg.min_sigma
            )
            threshold_high = noise_mu + cfg.thresh_std * noise_sigma
            threshold_low = noise_mu + cfg.hyst_ratio * cfg.thresh_std * noise_sigma
            threshold_high_parts.append(np.full(len(envelope), threshold_high, dtype=np.float32))
            threshold_low_parts.append(np.full(len(envelope), threshold_low, dtype=np.float32))
            if block_end >= calibration_samples:
                armed = True
            continue

        threshold_high = noise_mu + cfg.thresh_std * noise_sigma
        threshold_low = noise_mu + cfg.hyst_ratio * cfg.thresh_std * noise_sigma
        threshold_high_parts.append(np.full(len(envelope), threshold_high, dtype=np.float32))
        threshold_low_parts.append(np.full(len(envelope), threshold_low, dtype=np.float32))

        # Save any pending event only after the requested post-event audio exists.
        if pending_event is not None:
            enough_post_audio = block_end >= pending_event.trigger_sample + post_samples
            reached_file_end = block_end >= len(audio)
            if enough_post_audio or reached_file_end:
                record = save_event(
                    pending_event,
                    audio,
                    fs,
                    cfg,
                    output_dir,
                    metadata_file,
                    event_id,
                )
                records_for_file.append(record)
                event_id += 1
                pending_event = None

        if not in_event:
            can_trigger = (block_end - last_event_sample) > refractory_samples
            if score > threshold_high and can_trigger:
                trigger_sample = block_start + peak_idx
                last_event_sample = trigger_sample
                in_event = True
                pending_event = PendingEvent(
                    event_id=event_id,
                    source_file=str(wav_path),
                    trigger_sample=trigger_sample,
                    score=score,
                    peak=peak,
                    threshold_high=threshold_high,
                    threshold_low=threshold_low,
                    noise_mu=noise_mu,
                    noise_sigma=noise_sigma,
                )
                print(
                    f"[{wav_path.name}] event {event_id}: "
                    f"t={trigger_sample / fs:.3f}s score={score:.3e} "
                    f"thr={threshold_high:.3e}"
                )
            else:
                noise_mu, noise_sigma = update_noise_model(
                    noise_mu, noise_sigma, med, sigma_est, alpha, cfg.min_sigma
                )
        else:
            if score < threshold_low:
                in_event = False

    # End of file: save pending event if one was detected near the end.
    if pending_event is not None:
        record = save_event(
            pending_event,
            audio,
            fs,
            cfg,
            output_dir,
            metadata_file,
            event_id,
        )
        records_for_file.append(record)
        event_id += 1

    if make_plots:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        envelope_trace = np.concatenate(envelope_trace_parts) if envelope_trace_parts else np.array([])
        thresholds_high = np.concatenate(threshold_high_parts) if threshold_high_parts else np.array([])
        thresholds_low = np.concatenate(threshold_low_parts) if threshold_low_parts else np.array([])
        save_debug_plot(
            audio,
            envelope_trace,
            thresholds_high,
            thresholds_low,
            fs,
            records_for_file,
            wav_path,
            plot_dir,
        )

    return event_id, records_for_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline event detector for saved WAV files.")
    parser.add_argument("--input", required=True, type=Path, help="WAV file or directory of WAV files")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for event clips/metadata")
    parser.add_argument("--plots", action="store_true", help="Save debug waveform/envelope plots")
    parser.add_argument("--thresh-std", type=float, default=6.0, help="Detection threshold in robust sigmas")
    parser.add_argument("--hp-cutoff", type=float, default=150.0, help="High-pass cutoff frequency in Hz")
    parser.add_argument("--calibration-sec", type=float, default=3.0, help="Initial baseline calibration seconds per file")
    parser.add_argument("--pre", type=float, default=0.5, help="Seconds before trigger to save")
    parser.add_argument("--post", type=float, default=1.5, help="Seconds after trigger to save")
    parser.add_argument("--blocksize", type=int, default=1024, help="Processing block size in samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = DetectorConfig(
        blocksize=args.blocksize,
        thresh_std=args.thresh_std,
        hp_cutoff_hz=args.hp_cutoff,
        calibration_sec=args.calibration_sec,
        clip_pre_sec=args.pre,
        clip_post_sec=args.post,
    )

    wav_files = list(iter_wav_files(args.input))
    if not wav_files:
        print(f"No .wav files found in {args.input}")
        return

    metadata_path = args.output / "events.jsonl"
    summary_records: list[dict] = []
    next_event_id = 1

    print(f"Found {len(wav_files)} WAV file(s).")
    print(f"Writing clips and metadata to: {args.output}")

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for wav_path in wav_files:
            next_event_id, records = analyze_file(
                wav_path,
                args.output,
                metadata_file,
                cfg,
                next_event_id,
                args.plots,
            )
            summary_records.extend(records)

    print("\nDone.")
    print(f"Detected {len(summary_records)} event(s).")
    print(f"Metadata: {metadata_path}")
    if args.plots:
        print(f"Plots: {args.output / 'plots'}")


if __name__ == "__main__":
    main()
