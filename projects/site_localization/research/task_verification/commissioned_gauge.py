"""Conventional, local known-asset calibration. No reference-reading labels.

Angles are clockwise in reference image pixels, x right/y down. H maps reference
to query. The existing detector inverse-warps query pixels before measuring.
An accepted reading is conditional on asset association and the planar model;
it is NOT evidence of identity, physical pressure accuracy, or mission success.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def angle(point, center):
    delta = np.asarray(point, float) - center
    return math.degrees(math.atan2(delta[1], delta[0])) % 360


def calibrate(center, marks):
    """Exactly two independently observed ticks, never needle/reference values."""
    if len(marks) != 2:
        raise ValueError('Exactly two commissioned marks required')
    low, high = marks
    a = angle(low['point'], center)
    sweep = (angle(high['point'], center) - a) % 360
    if not 0 < sweep < 180 or high['value'] <= low['value']:
        raise ValueError('Require increasing values on a local clockwise arc <180deg')
    return {'angle_low_deg': a, 'angle_high_deg': a+sweep,
            'value_low': low['value'], 'value_high': high['value'],
            'value_per_deg': (high['value']-low['value'])/sweep}


def value_at(theta, calibration):
    """Local branch nearest the commissioned arc; no extrapolated acceptance."""
    c = calibration
    mid = (c['angle_low_deg']+c['angle_high_deg'])/2
    unwrapped = mid + (theta-mid+180) % 360 - 180
    value = c['value_low']+(unwrapped-c['angle_low_deg'])*c['value_per_deg']
    return float(value), c['angle_low_deg'] <= unwrapped <= c['angle_high_deg']


class CommissionedGauge:
    def __init__(self, bundle):
        self.bundle = Path(bundle)
        manifest = json.loads((self.bundle/'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if sha(self.bundle/name) != digest:
                raise ValueError('Frozen bundle mismatch: '+name)
        if sha(__file__) != manifest['files']['commissioned_gauge.py']:
            raise ValueError('Running implementation differs from frozen bundle')
        self.config = json.loads((self.bundle/'commissioning.json').read_text())
        self.reference = cv2.imread(str(self.bundle/'reference.png'), cv2.IMREAD_GRAYSCALE)
        spec = importlib.util.spec_from_file_location('commissioned_frozen_detector', self.bundle/'frozen_reader.py')
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        self.read_needle = reader.read_needle
        self.sift = cv2.SIFT_create()
        yy, xx = np.indices(self.reference.shape)
        cx, cy = self.config['detector']['center']
        radius = self.config['detector']['radius']
        rr = np.hypot(xx-cx, yy-cy)/radius
        m = self.config['registration']
        mask = ((rr >= m['ring_min']) & (rr <= m['ring_max'])).astype('uint8')*255
        self.keypoints, self.descriptors = self.sift.detectAndCompute(self.reference, mask)

    def read(self, image):
        result = {'decision': 'abstain', 'asset_identity_verified': False,
                  'physical_completion_verified': False,
                  'scope': 'conditional known-asset reading; planar transfer unvalidated physically'}
        if image is None or image.ndim != 2:
            return dict(result, reason='invalid_grayscale_image')
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if self.descriptors is None or descriptors is None:
            return dict(result, reason='no_features')
        settings = self.config['registration']
        pairs = cv2.BFMatcher().knnMatch(self.descriptors, descriptors, k=2)
        good = [p[0] for p in pairs if len(p)==2 and p[0].distance < settings['ratio']*p[1].distance]
        # A repeated tick must not count as several independent matches.
        unique = {}
        for match in sorted(good, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        good = list(unique.values())
        result['matches'] = len(good)
        if len(good) < settings['min_matches']:
            return dict(result, reason='insufficient_matches')
        src = np.float32([self.keypoints[m.queryIdx].pt for m in good])
        dst = np.float32([keypoints[m.trainIdx].pt for m in good])
        cv2.setRNGSeed(settings['ransac_seed'])
        h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, settings['ransac_px'])
        n = 0 if inliers is None else int(inliers.sum())
        result['inliers'] = n
        if h is None or not np.all(np.isfinite(h)) or n < settings['min_inliers']:
            return dict(result, reason='insufficient_inliers')
        if abs(np.linalg.det(h)) < 1e-12:
            return dict(result, reason='singular_homography')
        use = inliers.ravel().astype(bool)
        hull = cv2.convexHull(src[use])
        center = tuple(float(v) for v in self.config['detector']['center'])
        result['inlier_hull_area_px2'] = float(cv2.contourArea(hull))
        if cv2.pointPolygonTest(hull, center, False) < 0:
            return dict(result, reason='registration_does_not_surround_pivot')
        projected = cv2.perspectiveTransform(src[use][None], h)[0]
        result['inlier_reprojection_rms_px'] = float(np.sqrt(np.mean(np.sum((projected-dst[use])**2, axis=1))))
        result['reference_to_query'] = h.tolist()
        detector = self.config['detector']
        raw = self.read_needle(image, h, detector)
        # Old global value remains in diagnostics only; it is never the returned measurement.
        result['unchanged_detector'] = raw
        if raw['decision'] != 'accept':
            return dict(result, reason=raw['reason'])
        theta = detector['scale_start_deg']+raw['reading_fraction']*detector['scale_sweep_deg']
        value, inside = value_at(theta, self.config['calibration'])
        result['needle_angle_reference_deg'] = theta
        if not inside:
            return dict(result, reason='outside_commissioned_range', candidate_value=value)
        return dict(result, decision='accept', reason='conditional_calibrated_reading',
                    reading_value=value, units=self.config['units'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve previous output: '+str(args.out))
    cv2.setNumThreads(1)
    reader = CommissionedGauge(args.bundle)
    result = reader.read(cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE))
    result['image_sha256'] = sha(args.image)
    result['bundle_manifest_sha256'] = sha(args.bundle/'manifest.json')
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
