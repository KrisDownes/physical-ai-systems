"""Replay external frames through the unchanged moved-asset pixel detector.

This is component replay only: no robot pose, Gazebo truth, world location,
navigation result or independent reading label is available or inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
DETECTOR = ROOT / 'research/autonomous_asset/vision.py'
FREEZE = ROOT / 'results/fresh_inspection_aware_20261028/pipeline_freeze_v2.json'
DETECTOR_KEY = 'projects/site_localization/research/autonomous_asset/vision.py'
sys.path.insert(0, str(DETECTOR.parent))
from vision import detect


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def check_frame(frame, size):
    if frame is None:
        raise ValueError('Image could not be decoded')
    height, width = frame.shape[:2]
    if [width, height] != size:
        raise ValueError(f'Frame is {width}x{height}; frozen intrinsics require {size[0]}x{size[1]}. '
                         'Calibrate a new config before using another resolution; no automatic resize.')


def main(args):
    expected = json.loads(FREEZE.read_text())['files'][DETECTOR_KEY]
    if sha(DETECTOR) != expected:
        raise RuntimeError('Frozen detector hash differs from the inspected pipeline freeze')
    source = args.image if args.image is not None else args.video
    if not source.is_file():
        raise FileNotFoundError(source)
    config = json.loads(args.config.read_text())
    size = config['image_size']
    cv2.setNumThreads(1)
    if args.video is not None:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f'Video could not be opened: {source}')
        ok, first = cap.read()
        if not ok:
            cap.release()
            raise ValueError('Video has no decodable first frame')
        try:
            check_frame(first, size)
        except Exception:
            cap.release()
            raise
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        cap = None
        first = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        check_frame(first, size)
        fps = None
    args.out.mkdir(parents=True, exist_ok=False)
    frames = args.out / 'frames'
    frames.mkdir()
    manifest = dict(scope='offline external-frame detector replay; not a closed-loop or accuracy result',
                    source=str(source.resolve()), source_sha256=sha(source),
                    config=str(args.config.resolve()), config_sha256=sha(args.config),
                    unchanged_detector=str(DETECTOR), detector_sha256=expected,
                    detector_freeze=str(FREEZE.resolve()),
                    image_size=size, inspect=args.inspect,
                    calibration_note='Only pixel dimensions are checked. Pose requires matching real intrinsics, '
                                     'camera-to-robot geometry, and negligible distortion; the frozen config '
                                     'assumes a 60-degree ideal pinhole camera. Reading requires the same '
                                     'commissioned physical gauge. No ground-truth labels are provided.',
                    every_decoded_frame=args.every if cap is not None else None,
                    max_processed_frames=args.max_frames if cap is not None else 1,
                    video_reported_fps=fps, frames=[])

    def process(frame, index, video_ms):
        check_frame(frame, size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        name = f'frame_{index:06d}'
        png = frames / f'{name}.png'
        if not cv2.imwrite(str(png), gray):
            raise OSError(f'Could not write {png}')
        prediction = detect(gray, config, args.inspect)
        save(frames / f'{name}.json', prediction)
        manifest['frames'].append(dict(decoded_frame_index=index,
                                       video_timestamp_ms=None if video_ms is None else float(video_ms),
                                       saved_png=str(png.relative_to(args.out)),
                                       png_sha256=sha(png),
                                       prediction_json=str((frames/f'{name}.json').relative_to(args.out)),
                                       identity_decision=prediction['identity_decision'],
                                       observed_ids=prediction.get('observed_ids', []),
                                       reading_decision=prediction.get('reading', {}).get('decision')))

    try:
        process(first, 0, 0.0 if cap is not None else None)
        if cap is not None:
            index = 1
            while len(manifest['frames']) < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % args.every == 0:
                    process(frame, index, cap.get(cv2.CAP_PROP_POS_MSEC))
                index += 1
    finally:
        if cap is not None:
            cap.release()
        save(args.out / 'manifest.json', manifest)
    print(json.dumps(dict(processed=len(manifest['frames']), output=str(args.out),
                          decisions=[r['identity_decision'] for r in manifest['frames']]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=Path)
    group.add_argument('--video', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true',
                        help='Run the frozen commissioned gauge reader when ID7 is detected.')
    parser.add_argument('--every', type=int, default=30,
                        help='For video, process frame zero and every Nth decoded frame.')
    parser.add_argument('--max-frames', type=int, default=20,
                        help='Maximum processed video frames, including frame zero.')
    args = parser.parse_args()
    if args.every < 1 or args.max_frames < 1:
        parser.error('--every and --max-frames must be positive')
    main(args)
