"""Conventional image-only diagnostic. No simulation truth/maps/localization imports.

Reference matching is an OpenCV SIFT+RANSAC engineering baseline, not Spot's code.
The radial reading test is a simple known-gauge baseline, not a new contribution.
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


def decision(status, reason, **values):
    return {'decision': status, 'reason': reason, **values}


def read_needle(image, h, config):
    center = np.array(config['center'], float)
    radius = config['radius']
    # h takes the commissioned plane into this camera image.
    c = np.float32([center+[-radius, -radius], center+[radius, -radius],
                    center+[radius, radius], center+[-radius, radius]])
    face = cv2.perspectiveTransform(c[None], h)[0]
    diameter = min(np.linalg.norm(face[1]-face[0]), np.linalg.norm(face[3]-face[0]))
    values = {'face_corners': face.tolist(), 'diameter_px': float(diameter)}
    if (not np.all(np.isfinite(face)) or not cv2.isContourConvex(face) or
            np.any(face < [2, 2]) or np.any(face > [image.shape[1]-3, image.shape[0]-3]) or diameter < 80):
        return decision('abstain', 'insufficient_face_geometry', **values)
    rect = cv2.warpPerspective(image, np.linalg.inv(h), tuple(config['panel_shape']), borderValue=110)
    angles_deg = np.arange(config['scale_start_deg'], config['scale_start_deg']+config['scale_sweep_deg']+.01, .25)
    angles = np.radians(angles_deg)
    radii = np.linspace(.25*radius, .75*radius, 100)
    xs = (center[0] + np.cos(angles[:, None])*radii).astype(np.float32)
    ys = (center[1] + np.sin(angles[:, None])*radii).astype(np.float32)
    pixels = cv2.remap(rect, xs, ys, cv2.INTER_LINEAR)
    supports = np.mean(pixels < 100, axis=1)
    i = int(np.argmax(supports))
    # The scale spans 270 degrees, so no circular-endpoint alias in this band.
    competitor = float(np.max(supports[np.abs(angles_deg-angles_deg[i]) > 8]))
    fraction = float((angles_deg[i]-config['scale_start_deg'])/config['scale_sweep_deg'])
    values.update(peak_support=float(supports[i]), competing_support=competitor,
                  reading_fraction=fraction, reading_value=config['min_value']+fraction*(config['max_value']-config['min_value']))
    if supports[i] < .65 or supports[i]-competitor < .2:
        return decision('abstain', 'needle_not_uniquely_observed', **values)
    return decision('accept', 'face_and_needle_observed', **values)


class Verifier:
    def __init__(self, config, reference):
        self.config = config
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
        self.sift = cv2.SIFT_create()
        y, x = np.indices(reference.shape)
        cx, cy = config['center']
        dist = np.hypot(x-cx, y-cy)
        mask = ((dist > .82*config['radius']) & (dist < 1.06*config['radius'])).astype(np.uint8)*255
        self.keypoints, self.descriptors = self.sift.detectAndCompute(reference, mask)

    def reference_method(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if descriptors is None:
            return decision('abstain', 'no_features')
        pairs = cv2.BFMatcher().knnMatch(self.descriptors, descriptors, k=2)
        good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .7*b.distance]
        if len(good) < 10:
            return decision('abstain', 'insufficient_matches', matches=len(good))
        src = np.float32([self.keypoints[m.queryIdx].pt for m in good])
        dst = np.float32([keypoints[m.trainIdx].pt for m in good])
        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3)
        nin = 0 if mask is None else int(mask.sum())
        if h is None or nin < 8:
            return decision('abstain', 'insufficient_inliers', matches=len(good), inliers=nin)
        result = read_needle(image, h, self.config)
        result.update(matches=len(good), inliers=nin, panel_to_image=h.tolist())
        return result

    def marker_methods(self, image):
        corners, ids, _ = self.detector.detectMarkers(image)
        if ids is None:
            d = decision('abstain', 'no_tag')
            return d, dict(d)
        ids = ids.ravel().tolist()
        if ids.count(self.config['intended_tag_id']) != 1:
            d = decision('reject', 'wrong_or_nonunique_tag', observed_ids=ids)
            return d, dict(d)
        i = ids.index(self.config['intended_tag_id'])
        h = cv2.getPerspectiveTransform(np.float32(self.config['tag_corners']), corners[i][0])
        a = decision('accept', 'intended_tag_observed', observed_ids=ids)
        b = read_needle(image, h, self.config)
        b.update(observed_ids=ids, panel_to_image=h.tolist())
        return a, b


def main(inputs, out):
    assert not out.exists(), 'Preserve earlier decisions; choose a new output path.'
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20261023)
    config = json.loads((inputs/'commissioning.json').read_text())
    verifier = Verifier(config, cv2.imread(str(inputs/config['reference']), cv2.IMREAD_GRAYSCALE))
    results = []
    for row in json.loads((inputs/'samples.json').read_text()):
        image = cv2.imread(str(inputs/row['image']), cv2.IMREAD_GRAYSCALE)
        assert image is not None
        start, cpu = time.perf_counter(), time.process_time()
        ref = verifier.reference_method(image)
        ref_time = time.perf_counter()-start
        marker_start = time.perf_counter()
        tag, tag_read = verifier.marker_methods(image)
        results.append({'id': row['id'], 'arrival': decision('accept', 'controller_claim'),
                        'reference_reading': ref, 'tag_only': tag, 'tag_reading': tag_read,
                        'timing': {'reference_s': ref_time, 'tag_pair_s': time.perf_counter()-marker_start,
                                   'total_s': time.perf_counter()-start, 'cpu_s': time.process_time()-cpu}})
    out.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.inputs, a.out)
