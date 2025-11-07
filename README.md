# PJT

Ce dépôt contient des utilitaires pour entraîner les modèles Qwen sur les
fiches idées.  Le script principal se trouve dans `scripts/train_vision.py` et
permet de fiabiliser la préparation des données (conversion JSON → JSONL,
recalage des chemins d'images, extraction d'archives ZIP) avant le fine-tuning.

## Entraînement Vision

```bash
python scripts/train_vision.py \
  --dataset data/vision_dataset.jsonl \
  --image-zip data/antargaz_images_colab.zip \
  --image-dir dataset_fiches/vision \
  --output-dir models/fiche_idee_lora/vision
```

Le script accepte indifféremment un fichier JSON ou JSONL.  Si le fichier est
au format JSON, il est converti automatiquement en JSONL (format accepté par
`datasets.load_dataset`).  Toutes les images PNG du ZIP sont extraites dans le
répertoire indiqué afin que les chemins référencés dans le dataset soient
valides.  Si vous omettez l'argument `--dataset` ou `--image-zip`, le script
tente de les retrouver automatiquement dans `data/` ou `scripts/data/`.

### Exemple d'utilisation dans Google Colab

```python
!git clone https://github.com/<ton-compte>/PJT.git
%cd PJT

!pip install transformers accelerate datasets peft pillow bitsandbytes tqdm

# Téléverse ton dataset/ZIP si tu veux les remplacer
from google.colab import files
files.upload()  # vision_dataset.json(l) / antargaz_images_colab.zip

!python scripts/train_vision.py \
    --image-dir /content/PJT/dataset_fiches/vision \
    --output-dir /content/PJT/models/fiche_idee_lora/vision
```

Après l'upload, Colab place les fichiers dans `/content/`.  Le script journalise
les chemins résolus (`dataset`, `image_dir`, `image_zip`) pour vérifier que les
fichiers ont bien été trouvés avant de lancer le fine-tuning.

## Jeux de données d'exemple

Les dossiers `data/` et `scripts/data/` contiennent des jeux de données prêts à l'emploi qui corrigent
les problèmes relevés dans la version précédente :

* `vision_dataset.jsonl` : 13 fiches vision uniques, sorties JSON nettoyées et
  validées automatiquement.
* `context_dataset.jsonl` : 4 exemples d'analyse procédés couplant le rapport
  vision et le contexte texte.

Chaque fichier est encodé en JSON Lines (une entrée par ligne) et les champs
`output` respectent les attentes du script de fine-tuning (chaînes de caractères
contenant un JSON sérialisé pour la vision, texte enrichi suivi d'un verdict
JSON pour le contexte).
