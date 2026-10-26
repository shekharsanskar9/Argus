# Argus — Presentation Script & Cross-Question Prep

Everything you need to present `Argus_Presentation.pptx` with confidence.
Total talk time: **~15–18 minutes** (14 slides). Aim for ~1 to 1.5 min per slide,
leaving time for questions.

> **Tip:** Don't read the slides word for word. The slide shows the headline;
> *you* give the story. Below, each slide has a short "say this" script in your
> own voice. Practise it twice out loud.

---

## Part 1 — Speaker Script (slide by slide)

### Slide 1 — Title
> "Good morning. My project is called **Argus** — named after the hundred-eyed
> watchman from mythology. It's a computer-vision system that automatically
> detects motorcycle riders who aren't wearing a helmet, and then reads their
> number plate — from ordinary traffic images and video. I built it using
> YOLOv8, EasyOCR and OpenCV, and it runs on Google Colab."

*(Pause, then move on. ~30 sec.)*

### Slide 2 — Why it matters
> "Why did I build this? Three reasons. Helmets save lives — riding without one
> is a top cause of fatal head injuries. But manual enforcement doesn't scale —
> police can't watch every rider everywhere. Meanwhile, traffic cameras already
> record everything; nobody can review all that footage. So there's a clear gap
> that automation can fill — a system that watches 24/7 and flags violations
> consistently."

### Slide 3 — Objectives
> "My goal was a single pipeline with four jobs: **Detect** the objects in the
> scene, **Decide** whether a rider is violating, **Read** that rider's plate,
> and **Report** the result as annotated images and a table. Image in —
> violations and plate numbers out."

### Slide 4 — System architecture
> "Here's the whole pipeline. An image or video frame comes in. YOLOv8 detects
> riders, helmets and plates. Then the violation check — the amber box — decides
> who's breaking the rule. For each violator we crop the plate, straighten it
> and run OCR. Finally everything is written out as annotated images and CSV
> files. One nice detail: classes are matched by name, not by fixed numbers, so
> the same code works on any dataset with these labels."

### Slide 5 — Dataset
> "I used a public Kaggle dataset of real traffic scenes with four labelled
> classes: with-helmet, without-helmet, rider, and number-plate. Each class has
> a job in the pipeline — the 'without helmet' class is what triggers a
> violation, and 'number plate' is what gets sent to OCR. It's split into
> training and validation sets, in standard YOLO format with a data.yaml file."

### Slide 6 — Technology stack
> "These are the tools. **YOLOv8** from Ultralytics is the detector.
> **EasyOCR** reads the plate text, with PaddleOCR as a comparison.
> **OpenCV** does all the image processing — cropping, upscaling, straightening.
> It all runs on **Google Colab's free T4 GPU**, with PyTorch underneath and
> pandas for the result tables."

### Slide 7 — Training the detector
> "First step — detection. I started from the pre-trained yolov8n, the smallest
> 'nano' model, and fine-tuned it on my four classes for 50 epochs at 640-pixel
> resolution. On Colab's T4 GPU that trained in about two minutes. Then
> `model.val()` gives me precision, recall and mAP for each class."

*(Point at the code box.)* "This is literally the training call — just a few
lines thanks to Ultralytics."

### Slide 8 — Violation logic
> "Step two is the decision, and the rule is simple and explainable: a rider is
> a violation when the **centre of a 'without-helmet' box falls inside that
> rider's box** — that's the `center_in` function. On the right you can see it —
> the amber head-box's centre sits inside the cyan rider-box, so that's a
> violation. For evaluation I match my predictions to the ground truth using
> IoU — overlap — of at least 0.5."

### Slide 9 — OCR pipeline
> "Step three — reading the plate. I crop the plate, upscale it four to five
> times, and straighten it using Canny edge detection and Hough lines. Then,
> because OCR is fragile, I make several versions of the image — grayscale,
> sharpened, and two kinds of thresholding — run EasyOCR on each, and rank the
> results. I restrict it to letters and digits only, and I penalise words like
> HONDA or YAMAHA so the brand name doesn't get mistaken for the plate."

### Slide 10 — Detection results
> "Results. Overall the detector hit **0.92 precision and recall**, with an
> mAP50 of **0.92**. Looking per class, number plates are detected almost
> perfectly — 0.995. 'Without helmet' is the weakest at 0.81, which makes sense
> — heads are small and vary a lot. This is YOLOv8n after 50 epochs on just 20
> validation images."

### Slide 11 — Violation results
> "For the end-to-end violation test at confidence 0.35: it correctly found
> **9 of 11** real violations, with 2 false alarms and 2 misses — giving
> precision, recall and F1 all at **0.818**. I want to be honest here: with only
> 20 validation images these numbers are promising but noisy — a single image
> swings the score a lot."

### Slide 12 — Video extension
> "I also extended it to video. It runs the same detector on every frame, but
> adds **object tracking** so each rider gets a stable ID. That means a
> violating rider is counted **once**, not once per frame, and I keep the
> clearest plate reading I see for them across the whole clip. The output is an
> annotated video plus one row per violating rider."

### Slide 13 — Limitations & future work
> "I'll be upfront about the limits. The validation set is small, so metrics are
> noisy; OCR still struggles on blurry or angled plates; and the video tracker
> can swap IDs when riders overlap. For future work I'd train a larger model for
> longer, add a dedicated plate-recognition model with higher-resolution crops,
> and use a bigger, more varied dataset."

### Slide 14 — Conclusion
> "To conclude: Argus turns raw traffic footage into actionable
> helmet-violation records. One YOLOv8 model detects everything, the system
> flags violations and reads the plate automatically, it works on both images
> and video, and the whole thing is reproducible on free Colab GPUs. Thank you —
> I'm happy to take any questions."

---

## Part 2 — Cross-Question / Viva Prep

Questions a teacher is likely to ask, grouped by theme, with crisp answers.
Read the answer, then say it in your own words.

### A. Conceptual / "do you understand it"

**Q: What is YOLO and why did you choose it?**
YOLO = "You Only Look Once." It's a single-stage object detector that predicts
all bounding boxes and classes in one pass over the image, which makes it very
fast — fast enough for real-time video. I chose it because this is a
multi-object, real-time problem and YOLOv8 is accurate, well-supported, and easy
to fine-tune.

**Q: What does mAP mean?**
Mean Average Precision. For each class you plot precision against recall and take
the area under that curve (Average Precision); mAP is the mean across classes.
**mAP50** uses an IoU threshold of 0.5 to decide a correct box; **mAP50-95**
averages over thresholds 0.5 to 0.95, so it's stricter about box tightness.

**Q: What is IoU?**
Intersection over Union — the overlap area of two boxes divided by their combined
area. 1.0 is a perfect match, 0 is no overlap. I use IoU ≥ 0.5 to decide whether
a predicted box matches a ground-truth box.

**Q: What's the difference between precision and recall?**
Precision = of the violations I flagged, how many were real (penalises false
alarms). Recall = of the real violations, how many I caught (penalises misses).
F1 is their harmonic mean — a single balanced score.

**Q: How exactly do you decide a rider has no helmet?**
I don't compare helmet-vs-no-helmet on the same head directly. The model detects
a "without helmet" box; if its **centre lies inside a rider's bounding box**, I
attribute that bare head to that rider and flag a violation. That's the
`center_in` function.

**Q: Why is "fine-tuning" better than training from scratch?**
yolov8n is pre-trained on a huge general dataset (COCO), so it already knows
edges, shapes and common objects. Fine-tuning reuses that knowledge and only
adapts it to my four classes — so I get good accuracy from a tiny dataset in ~2
minutes instead of needing millions of images.

### B. Design decisions ("why did you…")

**Q: Why the nano model (yolov8n) and not a bigger one?**
It trains and runs fastest, which suited Colab's free GPU and my small dataset.
Accuracy was already strong. A bigger model (yolov8s/m) is my stated next step
for harder cases.

**Q: Why EasyOCR and not Tesseract or a custom model?**
EasyOCR is deep-learning based, handles real-world images better than classic
Tesseract, and needs no training. I also included PaddleOCR to compare. A
dedicated plate recogniser is future work.

**Q: Why all those image variants before OCR?**
Plates vary in lighting, blur and contrast. No single preprocessing wins every
time, so I generate grayscale, sharpened, Otsu and adaptive-threshold versions,
OCR each, and keep the best-scoring reading. It's a cheap way to boost
reliability.

**Q: Why penalise words like HONDA/YAMAHA?**
Those brand names are printed on the bike near the plate, and OCR sometimes reads
them with high confidence. Penalising known manufacturer words in my ranking
stops them from beating the actual plate number.

**Q: Why match classes by name instead of hard-coding IDs 0–3?**
So the code doesn't silently break if a dataset lists the classes in a different
order. `find_id("without helmet", …)` looks the ID up by name at runtime.

### C. Results & honesty

**Q: Your dataset only has 20 validation images — isn't that too small?**
Yes, and I say so on the limitations slide. The numbers are encouraging but
noisy — one image can swing the score by several percent. The honest conclusion
is that the *approach* works; proving the *exact* accuracy needs a much larger
test set, which is my first future-work item.

**Q: Which class performs worst and why?**
"Without helmet" (mAP50 ≈ 0.81). Heads are small, and bare heads look alike
across many poses and lighting conditions, so they're harder to localise than a
high-contrast rectangular number plate (0.995).

**Q: You had 2 false alarms and 2 misses — what causes those?**
Mostly the "without helmet" detector, plus the centre-inside rule: occlusion or
overlapping riders can put a head centre in the wrong rider's box, or miss it.
Better helmet detection and smarter rider-plate association would reduce both.

### D. Technical / environment

**Q: Why Colab and not your own laptop?**
My Mac has no NVIDIA CUDA GPU, and YOLOv8/EasyOCR run far faster on CUDA. Colab
gives a free T4 GPU, so training drops to ~2 minutes.

**Q: Could you run this on a TPU to go faster?**
No — and I checked. Ultralytics YOLOv8, EasyOCR and OpenCV are built on
PyTorch/CUDA and CPU; they don't use TPUs. A TPU runtime would sit idle while the
work fell back to the CPU. The T4 GPU is the right choice here.

**Q: Where are the outputs saved?**
On the Colab VM under `/content/` — `violations.csv`, `final_ocr_results.csv`,
the annotated images and the trained weights in `runs/detect/train/`. That
folder is temporary, so for permanent storage I'd mount Google Drive.

**Q: Is this real-time? What's the FPS?**
On a GPU, YOLOv8n detection is real-time (tens of FPS). The OCR step is the
bottleneck, but it only runs on frames that actually contain a violation, so the
tracker approach keeps it efficient by reading each rider's plate just a few
times, not every frame.

**Q: How does tracking know it's the same rider across frames?**
YOLO's tracker (with `persist=True`) assigns each detection a track ID and keeps
it consistent frame to frame using motion and position. I store the best plate
reading per track ID, so each rider is reported once.

### E. Extension / "what next" (shows maturity)

**Q: How would you deploy this in the real world?**
Feed a live CCTV/RTSP stream into the video pipeline, run it on an edge GPU or a
server, push violations with timestamp, plate and snapshot to a database, and
flag low-confidence plates for a human to verify before any action.

**Q: What about privacy / false accusations?**
Important point. The system should assist, not auto-penalise — a person reviews
flagged cases. Keeping OCR confidence with every reading, and showing the
cropped evidence image, lets a human make the final call.

**Q: What's the single biggest improvement you'd make?**
A larger, more varied dataset plus a dedicated plate-recognition model — that
would improve both the weak "without helmet" detection and the unreliable OCR,
which are my two main bottlenecks.

---

## Part 3 — Quick confidence checklist

- [ ] Know these 4 terms cold: **YOLO, mAP, IoU, precision/recall**.
- [ ] Be able to state the violation rule in one sentence (centre-inside).
- [ ] Remember the headline numbers: **0.92 overall mAP50**, **9/11 violations**,
      **~2 min training on a T4**.
- [ ] Have the honest limitation ready: **only 20 val images → noisy metrics**.
- [ ] Open the notebook/Colab beforehand in case the teacher asks for a live demo.
- [ ] Breathe, slow down, and point at the diagrams on slides 4 and 8.

Good luck — you built a genuinely solid end-to-end project. Present it like you
own it.
