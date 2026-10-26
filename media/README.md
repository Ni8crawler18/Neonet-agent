# media/

This repo ships **no video** — put your own clips here.

The default `feeds.json` expects two files:

- **`demo.mp4`** — a **static** beach camera (fixed viewpoint). A fixed camera is
  required: the OpenCV background-subtraction stage treats a moving/panning camera
  as motion everywhere. ~30–150 s is plenty.
- **`demo2.mp4`** — a sea / surf camera (open water, crashing waves) for the
  sea-state & rip-risk feed. ~30 s is enough.

Any ordinary `.mp4` works — e.g. a short recording from a public beach/coast webcam.
Name them `demo.mp4` and `demo2.mp4`, or edit `feeds.json` to point at your own
filenames. Then run `python selftest.py` to confirm both pipelines fire.
