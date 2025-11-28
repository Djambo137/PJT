# fiche-idee-global-vlm

Colab-first workflow to fine-tune **Qwen/Qwen2-VL-2B-Instruct** (LoRA) so multi-view fiche images + OCR text become strict CAD-ready JSON.

## Repo layout
```
.
├─ data/
│  ├─ vision_crops/                 # drop fiche crops here
│  └─ json_data/multiview_dataset.jsonl
├─ scripts/                         # OCR + dataset builders
├─ training/                        # dataset, collator, trainer, notebook
├─ inference/                       # predict + OpenSCAD scaffold
├─ metrics/                         # validation utilities
└─ outputs/                         # adapters + sample artifacts
```

## Data preparation
1. Export fiche views (PNG/JPG). Multi-views can use suffixes like `_a`, `_b`.
2. Resize images to keep the long side within **640–768 px** (scripts will also auto-resize).
3. Place them under `data/vision_crops/`.
4. (Optional) Use `scripts/ocr_and_split.py` for PDF→PNG, left/right splitting, and OCR (fra+eng).

### Build JSONL skeleton
```bash
python scripts/make_multiview_jsonl.py --images-dir data/vision_crops --output data/json_data/multiview_dataset.jsonl
python scripts/validate_jsonl.py data/json_data/multiview_dataset.jsonl
```
Each line groups the images for one fiche, keeps `context_text` empty by default, and embeds the fixed French instruction:

```
Tu es ingénieur produit. Analyse ensemble les IMAGES (vues/diagrammes) ET le TEXTE fourni. Rends un JSON STRICT et SEUL selon le schéma :
object, function, global_shape, components[], interfaces[], working_principle,
dimensions{H,W,D,unit}, materials_guess[], manufacturing[], assembly_hypothesis,
assumptions[], uncertainties[], risks_limits[]. Ne renvoie que le JSON valide.
```

Fill ~10–12 lines with gold `output` JSON (the nested object) before training. A tiny annotator is available inside the notebook (cell 4).

## Training on Colab (T4 friendly)
Open `training/finetune_colab.ipynb` (6 cells):
1. Install dependencies (bitsandbytes optional; falls back to FP16).
2. Clone the repo + install `requirements.txt`.
3. Data sanity check (prints count + previews 2 images).
4. Build JSONL skeleton, annotate inline if needed.
5. Train LoRA: epochs=5, batch=1, grad_accum=16, LR=1e-4, warmup_ratio=0.03, cosine schedule, fp16, gradient checkpointing.
6. Test one fiche then export OpenSCAD.

Trainer defaults target LoRA modules `q_proj,k_proj,v_proj,o_proj` and saves adapters to `outputs/qwen_lora_adapter/`.

### Common pitfalls
- **Image-token mismatch:** inputs preserve all image placeholders; `truncation=False` and two-pass encoding keep alignment.
- **OOM:** keep images near 640–768 px long side; batch size 1 with grad accumulation.
- **bitsandbytes missing:** warning prints, FP16 is used instead.

## Evaluation
- JSON validity rate: `python metrics/json_valid_rate.py predictions.jsonl`
- Slot F1 on key lists: `python metrics/slot_f1.py --gold data/json_data/multiview_dataset.jsonl --pred predictions.jsonl`

`predictions.jsonl` should contain one JSON object (model output) per line.

## Inference
```
python inference/predict_json.py --images path1 path2 ... --context "..." \
    --instruction default --adapter outputs/qwen_lora_adapter --output outputs/sample_pred.json
```
- Loads base `Qwen/Qwen2-VL-2B-Instruct`, applies adapter if provided, and validates output JSON against the schema.
- Exits non-zero on validation failure.

## Export to OpenSCAD
```
python inference/json_to_openscad.py outputs/sample_pred.json --out outputs/sample.scad
```
Emits a minimal cube/cylinder scaffold using `dimensions` and lists components/interfaces in comments. The `.scad` is syntactically valid and ready for refinement.
