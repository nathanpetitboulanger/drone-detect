# Données — détection de drones (1 classe `drone`, format YOLO)

Tout est reproductible par les scripts de `scripts/data/` : téléchargement → conversion YOLO →
découpage **par séquence** → statistiques → vidéos de démo. Les données ne sont pas versionnées
(`data/` est ignoré par git) ni redistribuées par ce dépôt : chacun les télécharge à la source.

- Config Ultralytics : **`configs/drones.yaml`** → `data/yolo/{train,val,test}/`.
- Contenu : Halmstad + DUT Anti-UAV tracking (découpés par séquence) + DUT Anti-UAV détection
  (en train seulement).
- Taille : **19 495 / 1 824 / 2 326 images** (train / val / test).

Historique : une première version (« v0 », sans DUT détection, 9 827 images en train) a servi à
valider la chaîne. Ses val et test sont identiques fichier pour fichier à ceux de la version
actuelle.

## Sources

| Source | Lien | Licence | Téléchargé | Séquences | Images retenues |
|---|---|---|---|---|---|
| Halmstad « Drone-detection-dataset » (Svanström et al.), vidéos visibles | [GitHub](https://github.com/DroneDetectionThesis/Drone-detection-dataset), [Zenodo 10.5281/zenodo.5500576](https://doi.org/10.5281/zenodo.5500576) | **CC0-1.0** (dépôt GitHub) ; Zenodo : « Other (Open) » | 311 Mo (archive complète), 125 Mo gardés (visible) | 285 vidéos 640×512 à 30 fps : 114 drones, 51 oiseaux, 59 avions, 61 hélicos (1 avion ignoré) | 9 015 |
| DUT Anti-UAV, partie tracking (Zhao et al., IEEE T-ITS 2022) | [GitHub](https://github.com/wangdongdut/DUT-Anti-UAV) (Google Drive) | **Apache-2.0** (licence du dépôt, pas de licence propre aux données) | 3,6 Go | 20 séquences 720p / 1080p, 24 804 frames | 4 962 |
| DUT Anti-UAV, partie détection | idem | **Apache-2.0** (idem) | 1,4 Go | aucune (10 000 images, découpage officiel train/val/test) | 9 668 (train seulement) |

Téléchargement total ≈ 5,3 Go, sans compte ni clé API. Il faut citer les articles des auteurs
(liens ci-dessus) en cas de publication.

**Sources écartées** :
- **Anti-UAV300 (RGB-T, licence MIT)** : 5,6 Go en Google Drive pour du RGB + IR. Le budget de
  10 Go serait dépassé. C'est une piste pour une v2 (seul le RGB serait utilisé).
- **Anti-UAV410 / 600** : infrarouge seulement.
- **Drone-vs-Bird** : accès sur dossier.
- **Roboflow Universe** : téléchargement par clé API (compte requis).
- **MAV-VID, Det-Fly** : pas vérifiés en détail, faute de temps. Les trois sources retenues
  couvraient déjà le besoin (séquences + distracteurs).

Remarques licences :
- Dans Halmstad, quelques clips avion / hélicoptère viennent d'une chaîne YouTube, d'après le
  fichier de description des auteurs. Ce sont des négatifs pour nous, publiés par les auteurs
  sous la licence du jeu.
- DUT Anti-UAV : le dépôt est sous Apache-2.0, ce qui implique attribution et conservation de la
  licence en cas de redistribution. Nous ne redistribuons pas les images.

## Conversion YOLO

- Une seule classe : `0: drone`. Labels `cls cx cy w h` normalisés. Un fichier vide marque une image
  négative. Les boîtes sont découpées aux bords de l'image ; celles de moins de 2 px sont ignorées.
- Nommage traçable : `<source>__<sequence>__<frame>.jpg`, avec la frame sur 6 chiffres,
  0-indexée (ex. `halmstad__V_DRONE_035__000100.jpg`).
- Chaque source a un manifeste `data/yolo/manifests/<source>.csv`, avec une ligne par image :
  séquence, groupe anti-fuite, frame, nb de boîtes, résolution, catégorie, attributs.

### Halmstad (`convert_halmstad.py`, `matlab_gt.py`)
- Les labels sont des objets MATLAB `groundTruth` (Video Labeler), que scipy ne sait pas lire.
  `matlab_gt.py` parcourt le flux MAT5 brut et retrouve la table des labels : par frame, une
  référence puis une cellule par classe (AIRPLANE, BIRD, DRONE, HELICOPTER), chacune de taille
  `[k, 4]`. Seule la colonne DRONE est utilisée, dans **toutes** les vidéos.
- **Pourquoi pas `mcos-decoder` (PyPI)** : il suppose une boîte par frame. Les 820 frames à
  **deux drones** (vidéos V_DRONE_106 à 108) sortent donc comme vides. Il ajoute aussi une entrée
  parasite en tête, l'en-tête « DRONE ». Notre décodeur trouve 36 316 boîtes drone, contre
  34 676 avec mcos-decoder, et autant de lignes de labels que de frames vidéo dans 284 vidéos sur
  285. Seule V_AIRPLANE_002 a un format différent ; elle est ignorée (vidéo sans drone).
- Convention MATLAB : coin haut-gauche 1-indexé, d'où −1 px.
- **Sous-échantillonnage** :
  - vidéos DRONE : 1 frame sur 5 (6 img/s, environ 60 images par clip de 10 s). Les frames
    voisines à 30 fps sont quasi identiques.
  - vidéos oiseau / avion / hélico : 1 frame sur 30 (1 img/s). Ce sont des **négatifs
    difficiles**, 21 % des images Halmstad. Oiseaux et aéronefs ne sont pas des drones.
- Contrôle visuel : `scripts/data/viz.py` génère des planches dans `outputs/data_viz/`.
  Les boîtes sont bien alignées, y compris sur les frames à 2 drones.

### DUT Anti-UAV tracking (`convert_dut_track.py`)
- GT : une ligne `x y w h` par frame. `-100 -100 -100 -100` signifie cible absente : ces frames
  sont gardées comme négatifs (10 % des frames).
- 1 frame sur 5. Les JPEG d'origine sont liés en dur, sans ré-encodage.
- FPS d'origine non documenté par les auteurs : 30 supposé pour les vidéos de démo.

### DUT Anti-UAV détection (`convert_dut_det.py`)
- VOC XML (`xmin ymin xmax ymax`, classe `UAV`), converti comme le script VOC d'Ultralytics.
  Aucun écart entre la taille déclarée dans l'XML et la taille réelle des 10 000 images.
- **Le découpage officiel train/val/test fuit.** Ces images sont des frames de vidéos. Sur une
  vignette globale (32×18), 65 % des images du test officiel ont une quasi-copie (similarité
  > 0,98) dans le train officiel, et 84 % des images train ont un quasi-doublon dans train.
  Une partie vient aussi des **mêmes décors que les séquences DUT tracking**, par exemple le
  bâtiment de video19, qui est une séquence démo.
- Choix retenu :
  - Les trois splits officiels sont fusionnés. La source sert **en train seulement**
    (`TRAIN_ONLY` dans `split.py`).
  - Chaque image est comparée à 1 frame sur 3 des 20 séquences tracking. Descripteur :
    vignette 64×36 en niveaux de gris passe-haut, similarité cosinus. Le passe-haut évite que
    deux ciels uniformes semblent identiques.
  - Au-delà de **0,5**, l'image est rattachée à la séquence (colonne `group = dut_track:<groupe>`).
    Le seuil a été fixé par contrôle visuel : 0,7 et plus = même décor, 0,3 à 0,55 = décors
    différents. Le nombre d'images liées à val/test reste à peu près constant entre 0,5 et 0,8
    (332 → 308).
  - Les images rattachées à une séquence val/test sont **écartées** (332), les autres vont en
    train (9 668).

## Découpage train / val / test (`split.py`)

- **Par groupe de séquences, jamais par image.** Graine fixe (`--seed 0`).
- Stratification par (source, catégorie), ce qui garantit des oiseaux, avions et hélicos dans
  chaque split. Cible ~70/15/15 en nombre d'images pour les sources découpées par séquence.
  Obtenu :
  - Halmstad : 69/14/17
  - DUT tracking : 73/11/16
- Groupes anti-fuite :
  - Halmstad : blocs de 5 clips consécutifs d'une même classe (`--group-size 5`). Les numéros
    voisins viennent souvent du même vol.
  - DUT tracking : une séquence = un groupe. Exception : video07+08 et video09+10, filmées au
    même endroit, restent ensemble.
- Les séquences démo sont forcées en test, avec tout leur groupe.
- Les splits sont des liens durs vers `data/yolo/all/`, reconstruits à chaque exécution.
  La table séquence → split est dans `data/yolo/splits.csv`.
- **Vérification** : `split.py --check` (aussi lancé par `split.py` et `stats.py`) lit les noms de
  fichiers de chaque split et vérifie qu'aucun couple `<source>__<sequence>` n'apparaît dans deux
  splits, et qu'aucun label ne manque. Résultat : OK.
- Ultralytics charge les trois splits sans fichier corrompu. Un
  `YOLO('yolo11n.pt').val(data='configs/drones.yaml', split='test', device='cpu')` tourne de bout
  en bout ; mAP ≈ 0, ce qui est normal : le modèle COCO n'a pas de classe drone.

| Split | Halmstad (séquences) | DUT tracking | DUT détection |
|---|---|---|---|
| train | 79 drone, 36 oiseau, 38 avion, 40 hélico | 13 séquences | 9 668 images |
| val | 15 drone, 10 oiseau, 10 avion, 10 hélico | video02, 05, 11, 16 | — |
| test | 20 drone, 5 oiseau, 10 avion, 11 hélico | video15, 18, 19 | — (332 écartées) |

## Statistiques (`stats.py`)

Taille des boîtes : côté équivalent √(w·h) en pixels, à la résolution d'origine. Ce sont les
mêmes seuils que COCO small / medium.

| split | séquences | images | boîtes | négatifs | < 16 px | < 32 px | < 96 px | médiane √(w·h) | médiane grand côté |
|---|---|---|---|---|---|---|---|---|---|
| **train** | 9874 | 19495 | 18060 | 1711 | 3% | 44% | 89% | 34.6 | 44.9 |
| train / dut_det | 9668 | 9668 | 9776 | 3 | 6% | 49% | 84% | 32.4 | 45.0 |
| train / dut_track | 13 | 3604 | 3147 | 457 | 0% | 1% | 88% | 62.9 | 88.0 |
| train / halmstad | 193 | 6223 | 5137 | 1251 | 0% | 60% | 100% | 30.4 | 34.4 |
| **val** | 49 | 1824 | 1496 | 328 | 2% | 51% | 86% | 31.4 | 34.8 |
| val / dut_track | 4 | 564 | 564 | 0 | 0% | 3% | 66% | 45.0 | 52.0 |
| val / halmstad | 45 | 1260 | 932 | 328 | 4% | 80% | 99% | 25.2 | 28.0 |
| **test** | 49 | 2326 | 1982 | 344 | 1% | 40% | 100% | 34.0 | 39.1 |
| test / dut_track | 3 | 794 | 736 | 58 | 0% | 13% | 100% | 45.1 | 53.0 |
| test / halmstad | 46 | 1532 | 1246 | 286 | 1% | 56% | 100% | 31.1 | 35.0 |
| **total** | 9972 | 23645 | 21538 | 2383 | 3% | 44% | 90% | 34.3 | 43.5 |

Pour DUT détection, « séquence » désigne une image.

| split | résolution : nb d'images |
|---|---|
| train | 1920×1080 : 10232, 640×512 : 6223, 1280×720 : 2762, 640×360 : 16, autres : 262 |
| val | 640×512 : 1260, 1920×1080 : 457, 1280×720 : 107 |
| test | 640×512 : 1532, 1920×1080 : 794 |

## Séquences de démo (`make_video.py` → `data/demo/`)

Ce sont des séquences de **test**, jamais vues à l'entraînement, reconstruites à pleine cadence
depuis les données brutes :

| Fichier | Contenu | Frames | Résolution |
|---|---|---|---|
| `data/demo/halmstad__V_DRONE_035.mp4` | F450, ciel couvert, cible petite (boîte ~25 px) | 302 @ 30 fps | 640×512 |
| `data/demo/halmstad__V_BIRD_020.mp4` | oiseaux proches (~2 par frame), **aucun drone** : test de fausses alarmes | 327 @ 30 fps | 640×512 |
| `data/demo/dut_track__video18.mp4` | ciel gris, petite cible au-dessus des arbres | 1 320 @ 30 fps (supposé) | 1920×1080 |
| `data/demo/dut_track__video19.mp4` | petite cible devant un bâtiment et le ciel | 1 300 @ 30 fps (supposé) | 1920×1080 |

**Hypothèse 30 fps pour DUT** : les auteurs ne publient que des JPEG, sans FPS d'origine. Les
vidéos DUT sont encodées à 30 fps (`--fps` pour changer) ; leur vitesse apparente peut donc
différer de la réalité. Les FPS Halmstad (30) sont ceux des vidéos d'origine.

`make_video.py --frames-dir <dossier> --out <x.mp4> --fps N` encode n'importe quel dossier d'images.

## Limites connues

- **Boîtes Halmstad parfois lâches** autour des cibles lointaines : une boîte de ~25 px peut
  entourer un drone de quelques pixels. La part de cibles < 16 px est donc sous-estimée, et le
  mAP50-95 sur Halmstad est pénalisé par l'imprécision du GT.
- Halmstad est en 640×512 et filmé sur un même site : décors peu variés, ciel majoritaire.
- Le test DUT ne compte que 3 séquences : la variance des scores par source est forte.
- Dans les 9 668 images DUT détection, beaucoup de frames voisines sont quasi identiques. Le
  train est donc moins varié que son nombre d'images ne le laisse penser.
- **Risque résiduel de fuite DUT** : le rattachement DUT détection → DUT tracking compare des
  vignettes globales. Il ne repère pas un même lieu filmé sous un angle ou un cadrage très
  différent. Des images d'entraînement DUT détection peuvent donc encore montrer le décor ou le
  drone des 3 séquences test DUT (video15, 18, 19), ce qui flatterait les scores sur ces
  séquences. Halmstad, majoritaire dans le test, n'est pas concerné. Pour une mesure prudente,
  regarder les scores par source.
- FPS des séquences DUT inconnu, 30 supposé (voir ci-dessus).
- Pas de séquence où un drone et un oiseau sont visibles en même temps.

## Reproduire (de bout en bout)

Depuis la racine du dépôt, avec l'environnement `.venv` (paquets de `requirements.txt`) et
`ffmpeg` (libx264) installés :

```bash
export YOLO_CONFIG_DIR=$PWD/.ultralytics
.venv/bin/python scripts/data/download.py --list          # sources, tailles, licences
.venv/bin/python scripts/data/download.py                 # ~5,3 Go → data/raw/, idempotent
.venv/bin/python scripts/data/convert_halmstad.py         # → data/yolo/all/…/halmstad
.venv/bin/python scripts/data/convert_dut_track.py        # → data/yolo/all/…/dut_track
.venv/bin/python scripts/data/convert_dut_det.py          # après dut_track (comparaison des décors)
.venv/bin/python scripts/data/split.py                    # train/val/test par séquence + vérif anti-fuite
.venv/bin/python scripts/data/stats.py                    # → data/yolo/stats.md
.venv/bin/python scripts/data/make_video.py               # → data/demo/*.mp4
.venv/bin/python scripts/data/viz.py --source halmstad --n 24   # contrôle visuel → outputs/data_viz/
```

Tous les scripts écrivent par défaut dans `data/yolo/`. `--yolo-dir <dossier>` construit une
copie ailleurs sans toucher au jeu utilisé par l'entraînement. Il faut alors passer le même
`--yolo-dir` à toutes les étapes et pointer un yaml sur ce dossier.

Durées sur la machine de dev : ~10 min de téléchargement (~10 Mo/s) ; conversion < 2 min ;
split + stats < 1 min.

Dépendances ajoutées pour ces scripts : `gdown` (Google Drive), `scipy` (lecture .mat),
`openpyxl` (fichier Excel de description Halmstad).
