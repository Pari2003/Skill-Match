# AI-Driven Job–Resume Matching with Experience-Aware Ranking

## Review Materials — Project Review Submission

---

## A. Problem Statement

### The Seniority-Blindness Problem in Semantic Job–Resume Matching

Modern AI-powered recruitment systems increasingly rely on **dense semantic embeddings** (e.g., Sentence-BERT) to match candidate resumes to job postings. These models encode textual content into high-dimensional vector spaces and rank candidates by cosine similarity. While effective at capturing topical relevance — matching skills, keywords, and domain vocabulary — **they are fundamentally blind to the candidate's experience level**.

#### Why Semantic Similarity Fails

Sentence-transformer models like `all-MiniLM-L6-v2` produce embeddings that reflect *what* a candidate has done, but not *how long* or *at what seniority*. Senior candidates accumulate more keywords, longer descriptions, and broader skill vocabularies over their careers. This creates a systematic bias: **Senior resumes consistently achieve higher cosine similarity scores regardless of whether the job requires senior-level experience.**

#### Measured Bias in Our Baseline

We evaluated this phenomenon on a dataset of **220 annotated resumes** matched against **499 curated job postings** across five industries (Healthcare, Finance, Technology, Manufacturing, Retail). Our baseline analysis reveals:

- **Top-1 Alignment Rate: 18.6%** — Only 1 in 5 top-ranked resumes matches the job's required experience level.
- **Senior Dominance: 58.4%** of all top-10 slots across ALL job tiers are occupied by Senior resumes, despite Seniors comprising 56.8% of the resume pool.
- **Fresher Under-exposure**: Even for jobs explicitly requiring Freshers, Senior resumes occupy 55.9% of top-10 slots.
- **Disparate Exposure Ratio (DER) ≈ 0.52** — Freshers receive roughly half the exposure of Seniors relative to their pool proportion.

#### The Need for Experience-Aware Ranking

This seniority bias has real-world consequences:

1. **Entry-level candidates are systematically deprioritised**, reducing their chances of being shortlisted even for appropriate positions.
2. **Overqualified candidates flood recommendations**, creating noise for recruiters.
3. **Purely skill-based corrections are insufficient** because skill YOE is correlated with career length.

We propose an **experience-aware ranking framework** that combines semantic similarity with (a) skill-specific experience scoring and (b) experience-level alignment modelling, to produce fairer and more relevant job–resume rankings.

---

## B. Methodology

### Overview

Our pipeline processes resumes and jobs through five progressive stages:

1. **Experience Extraction** — Structured parsing of resume annotations
2. **Baseline Semantic Matching** — Dense retrieval with all-MiniLM-L6-v2
3. **Bias Analysis** — Quantification of exposure disparity
4. **Skill-YOE Scoring** — Skill-specific experience factor
5. **Alignment Modelling** — Experience-level penalty/filter

### B.1 Experience Extraction Pipeline

From 220 annotated resumes (JSONL format with NER labels), we extract:
- **Work entries**: designation, company, date ranges
- **Duration computation**: month-level precision with overlap merging
- **Experience classification**:
  - Fresher: 0–18 months
  - Early-career: 19–36 months
  - Mid-level: 37–72 months
  - Senior: 73+ months

A reference date of 2024-01-01 is used for ongoing roles. Overlapping intervals (concurrent jobs) are merged to avoid double-counting.

### B.2 Baseline Semantic Matching

**Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings)

**Resume encoding**: For each resume, we concatenate role designations, company names, and extracted skills into a single text, then encode to a unit-normalised embedding.

**Job encoding**: Title and description are concatenated and encoded.

**Ranking**: Cosine similarity (computed as dot product of normalised vectors) produces a full 499×220 similarity matrix. Top-10 resumes per job are selected.

### B.3 Skill-YOE Scoring

**Skill Extraction**: We use `jjzha/jobbert_knowledge_extraction`, a BERT-based NER model fine-tuned for skill entity recognition, to extract required skills from job descriptions.

**Skill YOE Computation**: For each resume, skill-specific years of experience are computed at two confidence levels:
- **Explicit skills** (from resume annotations): attributed the resume's total overlap-merged experience
- **Inferred skills** (from designation NER): attributed per-role duration with a 0.5 weight discount

**Skill Experience Factor (SEF)**:

$$\text{SEF} = \frac{1}{|R|} \sum_{s \in R} \min\left(1, \frac{\text{YOE}_s}{24}\right)$$

where $R$ is the set of required skills and 24 months is the normalisation cap.

**Skill-Aware Score**:

$$\text{Score}_{\text{skill}} = \text{Similarity} \times (1 + \lambda \times \text{SEF})$$

with $\lambda = 0.3$.

### B.4 Hard Classification Filter

For each job with required experience class $C_j$:

1. Define allowed set $\mathcal{A} = \{C : |L(C) - L(C_j)| \leq 1\}$ where $L$ maps classes to ordinal levels (Fresher=0, Early-career=1, Mid-level=2, Senior=3)
2. Filter resumes to $\mathcal{A}$
3. Rank filtered candidates by cosine similarity

This guarantees no candidate more than one level away from the requirement appears in recommendations.

### B.5 Soft Alignment Penalty

Instead of hard filtering, apply a continuous exponential penalty:

$$\text{AlignmentFactor} = \exp(-\gamma \times |L_{\text{resume}} - L_{\text{job}}|)$$

$$\text{Score}_{\text{soft}} = \text{Similarity} \times \text{AlignmentFactor}$$

with $\gamma = 0.5$. This allows all candidates to compete but penalises level mismatches smoothly. A 1-level gap reduces the score by ~39%, a 2-level gap by ~63%, and a 3-level gap by ~78%.

### B.6 Combined Model

The final model integrates both skill-YOE and alignment:

$$\text{Score}_{\text{combined}} = \text{Similarity} \times (1 + \lambda \times \text{SEF}) \times \exp(-\gamma \times \text{gap})$$

with $\lambda = 0.3$, $\gamma = 0.5$.

This formula:
- Preserves semantic relevance as the base signal
- Rewards candidates with matching skill experience
- Penalises experience-level mismatches
- Re-ranks ALL 220 resumes per job (not just top-10 reorder)

---

## C. Results Summary

### Key Findings

- **Baseline semantic matching is seniority-blind**: 58.4% of all top-10 recommendations are Senior regardless of job tier. Top-1 alignment is only 18.6%.

- **Skill-YOE scoring alone has limited impact**: Due to low cross-domain vocabulary overlap (~5.7% between job and resume skill vocabularies) and the constraint of re-ranking only within the original top-10, class distribution remains unchanged at 58.4% Senior. Top-1 alignment shifts marginally to 18.2%.

- **Hard filtering dramatically improves alignment**: Top-1 alignment rises to **41.1%** and Senior dominance drops to **44.8%**. The model restricts candidates to matching ± adjacent classes, with an average of 120 candidates per job after filtering.

- **Soft alignment provides the strongest improvement**: Top-1 alignment reaches **99.2%** and NDCG@10 climbs to **0.9831**. Senior dominance drops to **19.6%**. For Fresher and Early-career jobs, Senior contamination falls to nearly 0%.

- **The combined model offers the best balance**: Top-1 alignment of **99.0%**, NDCG@10 of **0.9804**, Senior at **20.4%**. Integrating skill-YOE with alignment captures both: *what skills do you have?* and *is your experience level appropriate?*

### Trade-offs

| Approach | Strength | Weakness |
|----------|----------|----------|
| Baseline | Maximum semantic relevance | Seniority-blind, unfair exposure |
| Skill-YOE | Rewards skill-matched candidates | Limited by skill vocabulary overlap |
| Hard Filter | Strongest alignment guarantee | Reduced candidate pool, rigid |
| Soft Alignment | Smooth, continuous penalty | Does not reward skill relevance |
| Combined | Best of both: alignment + skill | Slightly more complex formula |

### Recommendation

**The Combined Model is recommended** as the primary ranking approach because it:
1. Maintains semantic relevance as the foundation
2. Explicitly rewards skill-experience match
3. Continuously penalises level mismatch without hard exclusion
4. Produces the best trade-off between alignment and diversity

---

## D. Architecture Diagram (Textual)

```
                    ┌──────────────────────────┐
                    │   Raw Annotated Resumes   │
                    │   (220 JSONL records)      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Experience Extraction     │
                    │  ├─ Work entry parsing     │
                    │  ├─ Date range extraction  │
                    │  ├─ Overlap merging        │
                    │  └─ Class assignment       │
                    │     (F / E / M / S)        │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────▼──────────┐      │      ┌────────────▼───────────┐
    │  Resume Embeddings  │      │      │  Skill-YOE Module      │
    │  (MiniLM-L6-v2)    │      │      │  ├─ Explicit skills     │
    │                     │      │      │  ├─ Inferred skills     │
    └─────────┬──────────┘      │      │  └─ Per-skill YOE       │
              │                  │      └────────────┬───────────┘
              │                  │                   │
              │    ┌─────────────▼──────────────┐    │
              │    │   Job Dataset (499 jobs)    │    │
              │    │   ├─ LinkedIn Postings      │    │
              │    │   ├─ Class mapping          │    │
              │    │   └─ Industry filtering     │    │
              │    └─────────────┬──────────────┘    │
              │                  │                   │
    ┌─────────▼──────────┐      │      ┌────────────▼───────────┐
    │  Cosine Similarity  │      │      │  JobBERT NER           │
    │  Matrix (499×220)   │      │      │  Required skills/job   │
    └─────────┬──────────┘      │      └────────────┬───────────┘
              │                  │                   │
              └──────────┬───────┴───────────────────┘
                         │
           ┌─────────────▼──────────────┐
           │     Scoring & Ranking       │
           │                             │
           │  Model 1: Baseline          │
           │    sim(job, resume)          │
           │                             │
           │  Model 2: Skill-YOE         │
           │    sim × (1 + λ×SEF)        │
           │                             │
           │  Model 3: Hard Filter       │
           │    Filter → sim             │
           │                             │
           │  Model 4: Soft Alignment    │
           │    sim × exp(−γ×gap)        │
           │                             │
           │  Model 5: Combined          │
           │    sim × (1+λ×SEF)          │
           │        × exp(−γ×gap)        │
           └─────────────┬──────────────┘
                         │
           ┌─────────────▼──────────────┐
           │     Evaluation              │
           │  ├─ Top-1 Alignment Rate    │
           │  ├─ NDCG@10                 │
           │  ├─ Exposure Metrics (DER)  │
           │  ├─ Class Distribution      │
           │  └─ Per-Tier Breakdown      │
           └─────────────┬──────────────┘
                         │
              ┌──────────▼──────────┐
              │  Ranked Results      │
              │  (Top-10 per job)    │
              └─────────────────────┘
```

### Pipeline Flow Summary

```
Resume JSON ──→ Experience Pipeline ──→ structured_experience_v2.json
                                              │
                                    ┌─────────┴──────────┐
                                    ▼                    ▼
                            Skill-YOE Module    Embedding Encoder
                                    │                    │
LinkedIn Jobs ──→ Job Preparation ──┤────────────────────┤
                                    │                    │
                                    ▼                    ▼
                            Skill Caches        Similarity Matrix
                                    │                    │
                                    └────────┬───────────┘
                                             ▼
                                    Ranking Models (×5)
                                             │
                                             ▼
                                    Evaluation Table
                                             │
                                             ▼
                                    Review-Ready Output
```

---

*Document prepared for Project Review — February 2026*
