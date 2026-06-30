"""Stage 6 — rally segmentation + hit detection (pure logic, image-space).

`segment.py` turns the per-frame shuttle spotter's detections into rallies: continuous-flight grouping,
foreign-court rejection by trajectory continuity, dropout bridging, serve/end cues, and a provisional
hit count. All trajectory logic runs in NORMALIZED IMAGE space (x=px/W, y=py/H) because the ground
homography can't place an airborne shuttle; court (u,v) is carried through only to expose the
floor-contact candidate for the future in/out feature.
"""
