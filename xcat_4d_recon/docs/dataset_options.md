# Dataset Options for 4D Dynamics Reconstruction

Ranked by match to the **fusion reactor (tokamak) setting**, where the observation
model is a fixed set of cameras placed around the torus, each producing a 2D
projection image of the plasma emission at each timestep.

## Key requirements

| Requirement | Fusion setting |
|---|---|
| **Observation type** | 2D projection images (line-of-sight integrated) |
| **Coverage** | Partial — fixed number of cameras (5–20), fixed angles |
| **Dynamics** | Quasi-periodic 3D volume evolving in time |
| **Inverse problem** | Reconstruct 3D(t) from limited 2D projections |
| **Ground truth** | Full 3D volume known only in simulation |

---

## Rank 1 — XCAT Phantom with Simulated DRR Projections ✓ (current setup)

**What it is:** The XCAT phantom provides 182 full 3D CT volumes of a breathing
chest. By generating Digitally Reconstructed Radiographs (DRRs) — simulated
parallel-beam X-ray projections — at a fixed set of angles, we directly mimic the
fusion camera observation model.

**Match to fusion:**

| Dimension | Match |
|---|---|
| Observation type | ✅ 2D projection images at fixed angles |
| Partial coverage | ✅ Configurable N cameras (5–8 angles) |
| 3D dynamics | ✅ Respiratory motion of chest/diaphragm |
| Quasi-periodic | ✅ Breathing is quasi-periodic (variable amplitude/phase) |
| Known ground truth | ✅ Full 3D volumes available (simulation) |
| Observation = emission | ⚠️ X-ray attenuation, not emission; physics differ but geometry is identical |

**Practical advantages:**
- Data already downloaded and preprocessed
- Full pipeline (PCA, delay embedding, MTTDE, DeepONet) already built around it
- DRR generation added to `preprocessing/generate_projections.py` — no new dependencies

**Download:** See main README. Data hosted at https://doi.org/10.5522/04/26132077.v1

---

## Rank 2 — SPARE Challenge: 4D CBCT

**What it is:** Cone-beam CT datasets from radiation therapy lung/liver treatment.
~900 2D X-ray projection images are acquired over ~1 minute while the patient
breathes freely. Only a sparse subset of projections (10–30) is available per
reconstructed phase.

**Match to fusion:**

| Dimension | Match |
|---|---|
| Observation type | ✅ Real 2D X-ray projections |
| Partial coverage | ✅ Sparse angular coverage, limited arc (~200° not full 360°) |
| 3D dynamics | ✅ Respiratory motion |
| Quasi-periodic | ✅ Free-breathing, variable cycles |
| Known ground truth | ✅ 4D-CT phase volumes available as reference |
| Data realism | ✅ Clinical scanner noise, scatter, motion blur |

**Why ranked below XCAT DRR:**
- Raw data format (projection files + geometry XML) requires non-trivial loading
- Scanner geometry (cone-beam) differs from parallel cameras
- Larger files, slower to iterate on

**Download:** https://sparechallenge.grand-challenge.org (registration required)

---

## Rank 3 — Cardiac Cine MRI (ACDC Dataset)

**What it is:** 150 patients, ~10 timepoints per cardiac cycle (end-diastole to
end-systole). 2D short-axis and long-axis MRI slices acquired from multiple
orientations.

**Match to fusion:**

| Dimension | Match |
|---|---|
| Observation type | ⚠️ 2D slices (not projections — MRI slices are thin, not line-of-sight sums) |
| Partial coverage | ⚠️ Only 2–3 fixed orientations, not a full set of angles |
| 3D dynamics | ✅ Cardiac contraction — strong 3D motion |
| Quasi-periodic | ✅ Heartbeat is highly periodic |
| Known ground truth | ✅ Full 3D volumes reconstructable from stack of slices |

**Why ranked below SPARE:**
- Observation model (slices) is not the same as camera projections
- Only 2–3 orientations, not the multi-angle ring geometry of fusion cameras
- MRI physics (spin relaxation) is very different from optical/X-ray

**Practical advantages:** Very clean NIfTI format, well-documented, many open-source
loaders, fast to download (~2 GB).

**Download:** https://acdc.creatis.insa-lyon.fr (free registration)

---

## Rank 4 — Dynamic PET (Total-Body PET)

**What it is:** Emission tomography where the scanner detects photons emitted by a
radiotracer injected into the patient. Multiple 2D sinogram projections are acquired
simultaneously over time.

**Match to fusion:**

| Dimension | Match |
|---|---|
| Observation type | ✅ Emission projections — closest physics to plasma emission cameras |
| Partial coverage | ✅ Full 360° but low count rates → effectively sparse |
| 3D dynamics | ✅ Tracer kinetics + respiratory motion |
| Quasi-periodic | ⚠️ Kinetics are not periodic — mostly monotone decay |
| Known ground truth | ⚠️ No clean GT; ground truth itself requires reconstruction |

**Why ranked lower despite best emission physics match:**
- Datasets are large, complex, and require specialized PET reconstruction software
- Kinetics are not quasi-periodic — Takens embedding assumptions are weaker
- Difficult to get started

---

## Summary table

| Dataset | Obs. type | Partial coverage | Quasi-periodic | Ease of use | Overall match |
|---|---|---|---|---|---|
| XCAT + DRR | 2D projection | ✅ configurable | ✅ breathing | ✅ ready now | ★★★★★ |
| SPARE 4D CBCT | 2D projection | ✅ sparse arc | ✅ breathing | ⚠️ complex format | ★★★★☆ |
| ACDC Cardiac MRI | 2D slice | ⚠️ few views | ✅ heartbeat | ✅ very clean | ★★★☆☆ |
| Dynamic PET | Emission proj. | ✅ full ring | ⚠️ not periodic | ❌ complex | ★★☆☆☆ |
