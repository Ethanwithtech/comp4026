Pins sample for anonymisation — COMP4026 Group 04
========================================================

Contents
--------
- images/                 315 face images (256x256 JPG) from 105 identities
- labels.csv              filename -> ground-truth identity

What to do
----------
1. Run your anonymisation model on every image in images/.
2. Save the output with the SAME FILENAME into a new folder named
   anonymised/ (e.g. images/Leonardo_DiCaprio_00.jpg
   -> anonymised/Leonardo_DiCaprio_00.jpg).
3. Keep the 256x256 JPG format (same as your CelebA-HQ outputs).
4. Send back the anonymised/ folder (zipped).

Notes
-----
- These are Pins Face Recognition test-set images (seed=42 split),
  so Student A's FaceRecognizer was originally trained on other
  images of the SAME identities. This lets us measure
  "Accuracy on anonymised images" cleanly, because the classifier
  is supposed to recognise these people.
- No identity labels are needed for your model to run — it only
  reads the image files. labels.csv is for Student A's evaluation.
