# Research Plan: COMP4026 Written Assignment

## Student Info
- Student ID: 22256342
- Course: COMP4026 Computer Vision and Pattern Recognition
- Due: 30 April 2026
- File naming: 22256342_Written_Assignment.pdf

## Assignment Structure
Two parts, 10 marks total:

### Part (i) - 5 marks
Select a YOLO-family algorithm proposed after 2018. Describe:
- Research problem it addresses
- Novelty (in own words)

**Selected algorithm: YOLOv7** (Wang, Bochkovskiy, Liao, CVPR 2023, arXiv:2207.02696)
- Reasons for selection:
  1. Strong CVPR paper, clear research problem statement
  2. Rich novelty: E-ELAN, model scaling for concatenation-based models, trainable bag-of-freebies (planned re-parameterization, deep supervision via auxiliary head with coarse-to-fine label assignment)
  3. Well-documented for academic writing
  4. Fits lecture timeline (between YOLOv3 and YOLO-World/YOLOE)

### Part (ii) - 5 marks
Essay (<400 words) on two unmet needs of SOTA visual object detection.

**Planned unmet needs:**
1. **True open-vocabulary generalization with reliable reasoning** - SOTA methods (even YOLO-World / YOLOE / DINO) still struggle with fine-grained, rare, or compositional concepts; they rely on CLIP-style alignment that is biased and brittle.
2. **Efficiency-accuracy-robustness trilemma on edge** - SOTA transformer detectors (DETR variants, Grounding-DINO) are accurate but heavy; YOLO variants are fast but degrade under domain shift, occlusion, small objects, adversarial conditions. No method simultaneously achieves SOTA accuracy, real-time edge latency, and robustness.

Alternative angles to consider:
- Data efficiency / few-shot learning (need huge labeled data)
- Explainability / interpretability
- 3D / multi-modal fusion
- Video temporal consistency
- Long-tail distributions

## Research Strategy
- Use WeChat article search skill + web research subagents in parallel
- Subagent 1: YOLOv7 deep dive (paper contents, novelty, research problem)
- Subagent 2: Unmet needs / limitations of SOTA object detection (post-2022)
- Subagent 3: Recent YOLO-World / YOLOE and transformer detector weaknesses

## Output
Final PDF: `/Users/yuchendeng/Desktop/comp4026/comp4026/22256342_Written_Assignment.pdf`
Via python-docx + PDF conversion (macOS: `textutil` or LibreOffice soffice).
