# COMP4026 Group 04 — Presentation Script
## Anonymised Facial Expression Recognition
**Target duration: ~7 minutes (≈ 1050 words at conversational pace)**

Speakers:
- **D** = Deng Yuchen (Face Recognition)
- **W** = Wang Xukun (Face Anonymisation)
- **Y** = Yi Tingxuan (Expression Recognition)

Approximate timings are printed in `[brackets]`. Stage directions are in *italics*.

---

## 🟦 SLIDE 1 — Title page
**Speaker: Deng Yuchen — `[0:00 – 0:20]`**

> Good afternoon, everyone. We are Group 04, and our project is
> **Anonymised Facial Expression Recognition**. I'm Deng Yuchen, and
> I worked on the face recognition model. My teammates are Wang Xukun,
> who built the anonymisation model, and Yi Tingxuan, who built the
> expression recognition model. Over the next seven minutes we'll show
> you how the three models fit together, and what the numbers actually
> say about whether our system protects privacy.

*→ advance to slide 2*

---

## 🟦 SLIDE 2 — Problem & Pipeline
**Speaker: Deng Yuchen — `[0:20 – 1:10]`**

> Face images are biometric data. Under GDPR they are a special
> category of personal data that requires explicit consent. But many
> real-world applications — retail analytics, classroom engagement,
> driver monitoring — only need the **expression**, not the identity.
>
> So our prototype does one simple thing: **we transform a face so its
> identity cannot be machine-matched, while the expression still
> survives**.
>
> The pipeline has three modules, shown on the right. A face image goes
> in. Student B's diffusion-based anonymiser rewrites the identity
> region. The output then gets evaluated in two directions: my face
> recognition model checks whether the original identity can still be
> recovered — that's the **privacy evaluator** — and Student C's
> expression classifier checks whether the expression is still
> recognisable — that's the **utility evaluator**. Each half measures
> one half of the project goal.
>
> With that context, let me start with the recognition model.

*→ advance to slide 3 (section divider)*

---

## 🟦 SLIDE 3 — Section: Face Recognition
**Speaker: Deng Yuchen — `[1:10 – 1:15]`** *(brief, basically a scene change)*

> Part 2 — the face recognition model.

*→ advance to slide 4*

---

## 🟦 SLIDE 4 — Student A: Architecture + Two-phase training + Results
**Speaker: Deng Yuchen — `[1:15 – 2:50]` (about 1.5 minutes)**

> The backbone is a **ResNet-50** pretrained on ImageNet. A 160-pixel
> face crop is projected down to a **512-dimensional L2-normalised
> embedding**, and an **ArcFace** loss with scale 32 and margin 0.3
> optimises angular separation between identities — much more
> discriminative than plain softmax for face tasks.
>
> The most interesting part of this work was actually a training
> failure. Our first attempt used ArcFace from epoch one with margin
> 0.5, the value from the original paper. On 105 identities the model
> simply would not converge: training accuracy stuck at zero,
> validation stuck at 42%. The reason is that the angular margin
> pushes the target logit so far down that the classifier can never
> predict the right class — so no useful gradient ever flows. A
> silent failure mode.
>
> The fix was a **two-phase schedule**. Epochs one to five use plain
> cross-entropy with label smoothing to shape the embedding space.
> Then at epoch six we switch to ArcFace with a gentler margin of 0.3,
> using differential learning rates.
>
> The four numbers at the bottom are the payoff.
> **Top-1 accuracy on the 105-identity Pins test set is 93.46%,
> Top-5 is 98.03%, EER on verification is 0.0528, and TAR at FAR
> of one-percent is 0.86**. That's a strong enough baseline to use
> as a privacy evaluator. Over to Xukun.

*→ advance to slide 5 (section divider)*

---

## 🟦 SLIDE 5 — Section: Anonymisation
**Speaker: Wang Xukun — `[3:10 – 3:15]`**

> Thanks, Yuchen. Part 3 — the anonymisation model.

*→ advance to slide 6*

---

## 🟦 SLIDE 6 — Student B: DDPM architecture & training
**Speaker: Wang Xukun — `[2:55 – 4:10]` (about 1.2 minutes)**

> My model is a **conditional Denoising Diffusion Probabilistic
> Model** with a U-Net denoiser, about 130 million parameters,
> running at 256×256 to match CelebA-HQ.
>
> The key design question was: how do you generate a *new* face that
> keeps the *original* expression, pose and gaze? Our answer is two
> conditioning signals. First, **468 facial landmarks from MediaPipe**
> lock in expression, head pose and gaze. Second, the **masked
> original image** preserves background. Both are injected into every
> U-Net block through **cross-attention**.
>
> We deliberately used **no identity loss** during training —
> adversarial identity losses tend to destabilise diffusion training
> and distort expression. Instead we train in a self-supervised way:
> L2 noise-prediction plus LPIPS for the first fifteen epochs, then
> progressive noise strength and classifier-free guidance in epochs
> sixteen through forty, while checking privacy against Yuchen's
> frozen FaceRecognizer. At inference we use **Poisson blending** to
> composite the generated face back into the original for a seamless
> result. Over to Tingxuan.

*→ advance to slide 7 (section divider)*

---

## 🟦 SLIDE 7 — Section: Expression Recognition
**Speaker: Yi Tingxuan — `[4:45 – 4:50]`**

> Thanks. Part 4 — expression recognition.

*→ advance to slide 8*

---

## 🟦 SLIDE 8 — Student C: architecture, training, results
**Speaker: Yi Tingxuan — `[4:15 – 5:10]` (about 55 seconds)**

> My task is a classifier that keeps working on anonymised faces.
> I use a **ResNet-50** with two adaptations: the first conv is
> modified for **1-channel grayscale input**, because FER-2013 is
> 48×48 grayscale; and the head is **Linear → BatchNorm → ReLU →
> Dropout 0.5 → Linear 7** for the seven expression classes.
>
> Training is two-stage. **For the first ten epochs the backbone is
> frozen** and only the head trains at learning rate 1e-3. **Then the
> whole network is unfrozen** with backbone at 1e-5 and head at 1e-4,
> plus cosine annealing and weight decay.
>
> Top-1 on FER-2013 is **68–70%** — the expected range for CNNs of
> this size, noting that human accuracy on FER-2013 is only 65-70%
> because the labels are noisy. On Student B's anonymised inputs
> accuracy drops by just a few points, to **65–68%**, and the
> **Expression Consistency Rate stays around 82-85%** — above the
> 80% project target. So utility is preserved. Back to Yuchen for
> the key evaluation.

*→ advance to slide 9*

---

## 🟦 SLIDE 9 — Joint Evaluation (the key slide)
**Speaker: Deng Yuchen — `[5:10 – 6:40]` (about 1.5 minutes)**

> This slide answers what the brief actually asks: **what is the
> accuracy on anonymised images?**
>
> We took **314 paired images from the Pins test split — the same
> 105 identities my classifier was trained on — ran them through
> Student B's anonymiser, and re-ran the classifier.** This is the
> clean comparison: one model, one identity space, one variable
> changing — whether the image went through the anonymiser.
>
> Top-1 on clean originals is **90%**, close to the 93% on the full
> test set, confirming the classifier still works on this sample.
> On anonymised images **Top-1 collapses to 17.5% — a drop of 72.6
> percentage points.** The anonymiser is clearly doing its job.
>
> But notice the third card: **random chance is 0.95%**. So 17.5%
> is still **eighteen times higher than chance**. Some identity-
> linked signal — pose, lighting, coarse geometry that the landmark
> conditioning is specifically asked to preserve — clearly survives.
>
> The histogram tells the same story in similarity space. Green is
> clean same-person pairs, centred around 0.6. Grey is impostor
> pairs, near zero. Orange is original-to-anonymised pairs at 0.30 —
> clearly shifted toward impostor, but not all the way. At our
> **EER-matched threshold of 0.35, 83% of clean genuine pairs pass
> verification, against only 37.6% of anonymised pairs — a Privacy
> Protection Rate of 62.4%**.
>
> Combined with the 80%-plus Expression Consistency Tingxuan showed,
> the pipeline hits both project targets — privacy drops sharply,
> utility is preserved — while making the residual leakage
> quantitatively visible.

*→ advance to slide 10*

---

## 🟦 SLIDE 10 — Contributions + Future Work
**Speaker: All three (one line each) — `[6:40 – 6:55]`**

> **D:** Yuchen — face recognition model and joint evaluation.
>
> **W:** Xukun — diffusion anonymiser with landmark and mask
> conditioning.
>
> **Y:** Tingxuan — grayscale expression classifier and consistency
> evaluation.
>
> **D:** The most promising next step is to train the anonymiser
> **with an explicit identity-dissimilarity loss against our frozen
> FaceRecognizer**, which would push the residual 37.6% verification
> rate down much further.

*→ advance to slide 11*

---

## 🟦 SLIDE 11 — Thanks / Q&A
**Speaker: Deng Yuchen — `[6:55 – 7:00]`**

> That's our prototype — **93% recognition, 62% privacy protection,
> 80%-plus expression consistency**. We'd be happy to take your
> questions.

*Open the floor.*

---

## 📋 ANTICIPATED Q&A (cheat sheet)

**Q1. Why did you pick ResNet-50 instead of a bigger backbone?**
> *D:* ResNet-50 is large enough to produce discriminative embeddings
> on 105 identities (24.6M parameters) but small enough to train fully
> on an Apple M-series GPU within a few hours. Given the size of the
> Pins training set (~12k images), a bigger model would mostly overfit.

**Q2. Why is 17.52% still so far from the 0.95% chance floor — can
you close that gap?**
> *D:* Three reasons. First, the landmark conditioning the anonymiser
> uses *intentionally* preserves pose and expression, which also
> preserves some coarse identity cues. Second, the anonymiser has no
> explicit identity-divergence loss during training, so it isn't
> directly optimised to fool a recogniser. Third, the residual Top-1
> is concentrated on near-frontal, neutrally-lit images. Our future
> work — adding an identity-dissimilarity loss — directly addresses
> points two and three.

**Q3. Why did you use EER-matched threshold 0.35 instead of 0.5?**
> *D:* 0.5 is the default, but it's not operating at the same point as
> our reported EER of 0.0528. Using the threshold that *produces* the
> EER gives a fair, consistent view of genuine vs. impostor behaviour.
> If we had used 0.5, the verification numbers would look artificially
> better because the cut would be stricter.

**Q4. Why didn't you use an adversarial identity loss in the
anonymiser?**
> *W:* Adversarial identity losses destabilise diffusion training and
> tend to distort expression landmarks. A cleaner approach is to
> train the diffusion model as a reconstruction task first, then tune
> privacy by sampling-time noise strength — and validate with a
> frozen recogniser. It's a design choice between stability and
> maximum privacy; we chose stability.

**Q5. What's the consistency rate when the anonymiser runs at
higher noise strength?**
> *W:* With the noise schedule set to its current setting, we get
> ~82-85% consistency. Pushing it further would drop consistency to
> maybe 75%, but drive residual verification down well below 30%.
> The parameter gives us a privacy/utility knob.

**Q6. How big is your training dataset and is 105 classes enough?**
> *D:* Pins is 17,534 images across 105 celebrity identities. For
> face recognition research this is small — papers often use
> MS-Celeb-1M with 10k identities — but for a prototype privacy
> evaluator it's sufficient. The two-phase trick we describe was
> actually *necessary* precisely because 105 classes is a regime
> where ArcFace margin 0.5 is too aggressive.

**Q7. Why does the anonymiser fail on some images?**
> *W:* Mostly on extreme profile views where MediaPipe returns fewer
> reliable landmarks. With weaker conditioning the U-Net has to
> hallucinate more geometry, which can preserve identity
> accidentally.

**Q8. Is the system GDPR-compliant?**
> *D:* Our goal was a research prototype that *measures* privacy
> protection, not a production GDPR solution. A deployed system would
> need additional safeguards — audited data handling, a stronger
> privacy bound from an adversarial audit, and a formal DPIA. But
> the pipeline design is consistent with privacy-by-design
> principles.

---

## 🧭 DELIVERY TIPS

1. **Pace**: 1050 words / 7 minutes ≈ 150 wpm. Conversational.
2. **Slide 7 is the headline slide.** If you are running short on
   time, cut the Q&A cheat-sheet answers, not the evaluation
   discussion.
3. Hand-offs are explicit ("I'll hand over to Xukun", "Tingxuan will
   talk about the expression side", "Yuchen will now show…"). Use
   them — it signals organisation.
4. When you mention a number on Slide 9, **look at the audience, not
   the screen**. The slide backs you up; you shouldn't be reading it.
5. Rehearse Slide 4 and Slide 9 at least three times each — they
   contain the densest content.
