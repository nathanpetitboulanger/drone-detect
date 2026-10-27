# Export ONNX / TensorRT et mesure des FPS

## En bref

| | R2 (démo), imgsz 1280 | R1 (léger), imgsz 640 |
|---|---|---|
| Exports | `best_1280.onnx` (11,2 Mo), `best_1280_fp16.engine` (9,2 Mo) | `best_640.onnx` (10,7 Mo), `best_640_fp16.engine` (8,1 Mo) |
| Vérification vs PyTorch (102 images) | ONNX et engine **OK** | ONNX **OK** ; engine : 1 boîte doublon décalée de 2 px (voir plus bas) |
| Inférence TensorRT FP16 seule | **2,56 ms** (391 FPS) | **1,02 ms** (979 FPS) |
| Boucle de démo complète, vidéo 1080p | **106 FPS** (TensorRT) / 72 FPS (PyTorch) | 186 FPS (TensorRT) / 95 FPS (PyTorch) / 21 à 27 FPS sur CPU seul |

Ces mesures ont été faites sur un PC portable (RTX 3070 Laptop, Ryzen 7 5800H), machine au
repos. Ce ne sont **pas** des chiffres Jetson.

Deux scripts :

- [`scripts/export.py`](../scripts/export.py) exporte des poids `.pt` en ONNX et en engine
  TensorRT, puis vérifie que le modèle exporté prédit les mêmes boîtes que PyTorch ;
- [`scripts/bench.py`](../scripts/bench.py) mesure latence et FPS en batch 1 pour une liste de
  poids (`.pt`, `.onnx`, `.engine`) et note le matériel et la charge de la machine.

L'artefact à déployer sur un Jetson est l'**ONNX**. Un engine TensorRT dépend du GPU et de la
version de TensorRT qui l'ont construit. On le **reconstruit donc sur le Jetson**. Les engines
construits ici ne servent qu'aux mesures locales et à la démo.

## Environnement

| Composant | Version | Rôle |
|---|---|---|
| ultralytics | 8.4.190 | `model.export(...)`, inférence, suivi |
| torch | 2.14.1 + CUDA 13.0 | référence PyTorch |
| onnx / onnxslim | 1.23.2 / 0.1.98 | export et simplification du graphe |
| onnxruntime-gpu | 1.31.1 | inférence ONNX (CPU, CUDA) |
| tensorrt-cu13 | 10.16.1.11 | construction et inférence des engines (roue pip NVIDIA, 3,7 Go téléchargés) |

Le choix de **TensorRT 10** plutôt que 11 est délibéré :

- les JetPack 6 et 7 embarquent TensorRT 10.x ;
- `onnxruntime-gpu` 1.31 est lié à `libnvinfer.so.10` ;
- TensorRT 11 n'a plus que des réseaux « fortement typés ». Pour le FP16, Ultralytics doit
  alors passer par NVIDIA ModelOpt, une dépendance supplémentaire qu'il installe tout seul.

Installation : `uv pip install --python .venv/bin/python "tensorrt-cu13==10.16.1.11"`.

## Export — `scripts/export.py`

Commandes utilisées (GPU) :

```bash
.venv/bin/python scripts/export.py --weights runs/train/r2_y11n_1280_ft/weights/best.pt --imgsz 1280 \
    --formats onnx engine --device 0 --check-n 100
.venv/bin/python scripts/export.py --weights runs/train/r1_y11n_640/weights/best.pt --imgsz 640 \
    --formats onnx engine --device 0 --check-n 100
```

Les fichiers sont écrits à côté des poids (ou dans `--out`) :

- `best_<imgsz>[_dyn].onnx` ;
- `best_<imgsz>[_dyn]_<fp16|fp32|int8>.engine` ;
- `best_export.json`, le rapport : taille, temps d'export, versions et vérification de chaque
  fichier.

Autres options :

- `--dynamic` : axes dynamiques (batch, hauteur, largeur) ;
- `--skip-existing` : réutiliser un export existant et ne refaire que la vérification ;
- `--precision fp32|int8` : précision de l'engine (l'INT8 est calibré sur `--data`) ;
- `--nms` : intégrer la NMS au modèle.

Réglages par défaut : batch 1, **opset 17** (lu par TensorRT 8.5 et plus, donc JetPack 5 et 6),
graphe simplifié par onnxslim, NMS **hors** du modèle, engine en FP16. La sortie brute du
modèle drones est un tenseur `1 × 5 × N` (4 coordonnées + 1 score), avec N = 8 400 à 640 et
N = 33 600 à 1 280.

### Vérification face à PyTorch

`export.py` fait prédire PyTorch FP32 et le modèle exporté sur les mêmes images, sur le même
device, avec la même letterbox carrée. Les images sont `bus.jpg`, `zidane.jpg` et `--check-n`
images réparties sur le split test ; ici 100, soit 102 images en tout.

Le script apparie ensuite les boîtes (même classe, IoU ≥ 0,5). Le verdict est **OK** si chaque
boîte « stable » a son homologue à **IoU ≥ 0,9** de l'autre côté. Une boîte est stable si sa
confiance dépasse le seuil de 0,25 d'au moins 0,05, soit ≥ 0,30. Le critère est volontairement
strict. En cas d'écart, le script sort avec le code 2 et le JSON détaille chaque image.

| Fichier | Taille | Export | Boîtes appariées | IoU moy. / min | Δ confiance max | Δ coordonnées max | Verdict |
|---|---|---|---|---|---|---|---|
| R2 `best.pt` (référence) | 5,6 Mo | — | — | — | — | — | — |
| R2 `best_1280.onnx` | 11,2 Mo | 2 s | 106 / 106 | 0,9999 / 0,9997 | 3,0e-4 | 0,007 px | **OK** |
| R2 `best_1280_fp16.engine` | 9,2 Mo | 194 s | 106 / 106 | 0,9984 / 0,9951 | 4,7e-3 | 0,09 px | **OK** |
| R1 `best.pt` (référence) | 5,5 Mo | — | — | — | — | — | — |
| R1 `best_640.onnx` | 10,7 Mo | 2 s | 131 / 131 | 0,9999 / 0,9994 | 2,2e-4 | 0,007 px | **OK** |
| R1 `best_640_fp16.engine` | 8,1 Mo | 222 s | 131 / 131 | 0,9973 / 0,8312 | 9,7e-3 | 2,1 px | ÉCART (1 boîte) |

**L'ÉCART de l'engine R1 est sans conséquence.** Il vient d'une seule image,
`halmstad__V_DRONE_038__000235.jpg` :

- PyTorch y produit deux boîtes sur le **même** drone : la détection principale (conf 0,433)
  et un doublon plus lâche (conf 0,323). La NMS les garde toutes les deux, car leur IoU mutuelle
  reste sous 0,7.
- En FP16, la détection principale est identique (écart < 0,1 px, conf 0,434). Le doublon,
  juste au-dessus du seuil « stable » de 0,30, se décale de 2 px. Comme la cible ne mesure
  qu'environ 30 × 20 px, son IoU tombe à 0,83.
- C'est du bruit d'arrondi FP16 sur une boîte ambiguë, pas une divergence du modèle. Toutes les
  boîtes restent appariées, et le verrouillage, qui suit la piste la plus confiante, n'est pas
  concerné.

On garde ce critère strict plutôt que de l'assouplir pour obtenir un « OK ».

L'ONNX pèse environ deux fois le `.pt`, ce qui est normal : Ultralytics stocke les poids du
`.pt` en FP16, alors que l'ONNX est en FP32.

## Mesure des FPS — `scripts/bench.py`

Commandes utilisées. Script : `runs/bench/final.sh` ; sorties : `runs/bench/final_*.json` et
`.md`.

```bash
W2=runs/train/r2_y11n_1280_ft/weights; W1=runs/train/r1_y11n_640/weights; V=data/demo/dut_track__video19.mp4
.venv/bin/python scripts/bench.py --weights $W2/best.pt $W2/best_1280.onnx $W2/best_1280_fp16.engine \
    --pt-precision fp32 fp16 --imgsz 1280 --device 0 --video $V --wait-idle 15 --name final_gpu_r2_1280
.venv/bin/python scripts/bench.py --weights $W1/best.pt $W1/best_640.onnx $W1/best_640_fp16.engine \
    --pt-precision fp32 fp16 --imgsz 640 --device 0 --video $V --wait-idle 15 --name final_gpu_r1_640
.venv/bin/python scripts/bench.py --weights $W1/best.pt $W1/best_640.onnx \
    --imgsz 640 --device cpu --video $V --wait-idle 15 --name final_cpu_r1_640
```

### Ce qui est mesuré

Batch 1, sur une image 1920×1080 du split test (`dut_track__video15__000000.jpg`) gardée en
mémoire. 20 prédictions de chauffe, puis 200 prédictions mesurées (médiane et p90).

| Colonne | Définition |
|---|---|
| **Inférence** → FPS modèle | Passe avant du réseau seule (`result.speed["inference"]`, synchronisée CUDA). |
| **Predict** → FPS predict | Appel `model.predict` complet sur l'image décodée : letterbox, normalisation, inférence, NMS et boîtes ramenées sur CPU. |
| **FPS vidéo** | Boucle de `demo.py` sans affichage ni écriture, sur `data/demo/dut_track__video19.mp4` (1920×1080, séquence test) : lecture de la frame + `model.track` (ByteTrack) + verrouillage (`lockon.py`). 300 frames après 10 de chauffe ; FPS = frames / temps total. C'est la cadence de la boucle de guidage. |

### Règles d'équité

- **Même entrée pour tous les formats.** La letterbox est carrée (imgsz × imgsz), comme pour
  un export à taille fixe. Avec `--rect`, PyTorch et l'ONNX dynamique prennent une entrée
  rectangulaire minimale : 1920×1080 → 640×384, donc moins de calcul. C'est le cas de `demo.py`
  avec un `.pt`.
- **ONNX Runtime sans attente active.** Par défaut, les threads d'ORT tournent à vide
  (*spinning*) après chaque inférence. Ils prennent alors le CPU aux threads PyTorch du
  pré/post-traitement d'Ultralytics, et les deux ralentissent. Pendant le développement, on a
  observé jusqu'à ×3 sur l'appel predict en CPU, sur machine chargée. `bench.py` et `demo.py`
  désactivent ce spinning (`tune_runtime` dans `bench.py`) ; `--ort-spinning` garde le
  comportement d'origine.
- **Machine au repos, et vérifiée.**
  - `bench.py` note le matériel, les versions, l'alimentation et la charge système au début de
    chaque run et à la fin de chaque ligne.
  - Pendant chaque mesure GPU, `nvidia-smi pmon` échantillonne l'activité des autres processus
    une fois par seconde. Une ligne mesurée pendant qu'un autre processus calculait sur le GPU
    est marquée †.
  - `--wait-idle 15` attend 15 s de GPU calme avant chaque ligne.
  - Sur cette machine partagée, des mesures prises pendant d'autres calculs (inférence CPU d'un
    autre projet, compilation, llama-bench) ont été **écartées et refaites** : les chiffres
    passaient alors de 31 à 63 ms en ORT CPU, par exemple.

### Résultats sur la machine locale

- **Matériel** : PC portable, NVIDIA GeForce RTX 3070 Laptop (8 Go, 115 W max, pilote
  615.78.08) et AMD Ryzen 7 5800H (8 cœurs, 16 threads, 8 threads torch), sur secteur,
  Arch Linux (noyau 7.2).
- **Logiciel** : torch 2.14.1 + CUDA 13.0, Ultralytics 8.4.190, ONNX Runtime 1.31.1,
  TensorRT 10.16.1.
- **Conditions** : 27/10/2026, 19:44 à 19:49.
  - Runs GPU : charge système (1 min) entre 2,2 et 3,6. Seul un navigateur ouvert consommait
    environ 2 cœurs. Aucune activité GPU d'un autre processus pendant aucune ligne (aucune
    ligne †).
  - Run CPU : charge de 2,6 au lancement, 4,8 à 5,8 en fin de ligne. Ce chiffre inclut les
    8 threads du benchmark lui-même.

| Modèle | Runtime | Précision | Entrée | Inférence méd. / p90 (ms) | FPS modèle | Predict méd. / p90 (ms) | FPS predict | FPS vidéo 1080p bout en bout |
|---|---|---|---|---|---|---|---|---|
| R2 `best.pt` | PyTorch CUDA | FP32 | 1280×1280 | 7,22 / 7,26 | 138 | 11,45 / 11,84 | 87 | 72,5 |
| R2 `best.pt` | PyTorch CUDA | FP16 | 1280×1280 | 6,22 / 6,78 | 161 | 10,40 / 11,05 | 96 | 76,5 |
| R2 `best_1280.onnx` | ONNX Runtime CUDA | FP32 | 1280×1280 | 7,92 / 8,03 | 126 | 12,52 / 12,97 | 80 | 67,8 |
| **R2 `best_1280_fp16.engine`** | **TensorRT** | **FP16** | 1280×1280 | **2,56 / 2,58** | **391** | **6,99 / 7,59** | **143** | **106,4** |
| R1 `best.pt` | PyTorch CUDA | FP32 | 640×640 | 6,05 / 8,14 | 165 | 8,34 / 11,12 | 120 | 95,0 |
| R1 `best.pt` | PyTorch CUDA | FP16 | 640×640 | 6,73 / 7,66 | 149 | 9,12 / 10,43 | 110 | 91,5 |
| R1 `best_640.onnx` | ONNX Runtime CUDA | FP32 | 640×640 | 2,71 / 2,99 | 369 | 4,89 / 5,45 | 204 | 131,4 |
| R1 `best_640_fp16.engine` | TensorRT | FP16 | 640×640 | 1,02 / 1,04 | 979 | 3,08 / 3,64 | 325 | 185,7 |
| R1 `best.pt` | PyTorch CPU | FP32 | 640×640 | 33,99 / 38,26 | 29 | 36,69 / 40,84 | 27 | 21,5 |
| R1 `best_640.onnx` | ONNX Runtime CPU | FP32 | 640×640 | 30,50 / 33,77 | 33 | 32,90 / 36,13 | 30 | 27,3 |

Lecture :

- **TensorRT FP16 divise par 2,8 à 6 le temps d'inférence** par rapport à PyTorch :
  - R2 à 1280 : 7,22 → 2,56 ms ;
  - R1 à 640 : 6,05 → 1,02 ms.
- **PyTorch CUDA est limité par le CPU sur ce modèle nano.** R1 à 640 (6,05 ms) n'est guère plus
  rapide que R2 à 1280 (7,22 ms), pour 4 fois moins de pixels. La passe avant est dominée par le
  lancement des quelque 100 couches depuis Python, pas par le calcul GPU. Pour la même raison, le
  FP16 PyTorch ne gagne rien à 640 (et un peu à 1280). Les runtimes compilés (ORT, TensorRT)
  n'ont pas ce coût.
- **Avec TensorRT, le réseau n'est plus le goulot.** Pour R2 à 1280, l'appel predict prend
  7 ms, dont 2,6 ms de réseau ; le reste se répartit ainsi :
  - letterbox et normalisation de l'image 1080p : 3,1 ms ;
  - NMS : 0,9 ms ;
  - surcoût Python.
  La boucle vidéo (décodage H.264 1080p + suivi + verrouillage) ajoute environ 2 ms, pour
  9,2 ms par frame, soit **106 FPS**. Le prochain gain viendrait d'un pré-traitement et d'un
  décodage sur GPU (voir DeepStream).
- **La démo réelle est un peu plus lente.** `demo.py` avec l'engine R2 tourne à 93 à 96 FPS sur
  les deux séquences DUT 1080p (« FPS pipeline » de la démo). Il écrit en parallèle la vidéo
  annotée et le GIF.
- **Sur CPU seul, R1 à 640 tient 21 à 27 FPS** de bout en bout. C'est utilisable pour une démo,
  mais trop juste pour R2 à 1280, qu'on n'a pas mesuré sur CPU.
- **ORT CUDA** est 2,2 fois plus rapide que PyTorch à 640, mais pas à 1280 (7,9 contre 7,2 ms).
  L'exécution CUDA d'ORT sans TensorRT reste moins optimisée que TensorRT.

Ces chiffres valent pour un GPU de portable de bureau. Ils ne disent pas ce que fera un Jetson,
dont le GPU est plusieurs fois moins puissant et le CPU ARM bien plus lent. Il faudra mesurer
sur la cible (voir plus bas).

## Déploiement sur Jetson (non testé, pas de Jetson disponible)

### Versions

| JetPack | TensorRT | Remarque |
|---|---|---|
| 5.1.x (Xavier, Orin) | 8.5 | opset 17 accepté |
| 6.0 (Orin) | 8.6 | |
| 6.1 / 6.2 (Orin) | 10.3 | 6.2 : mode « Super » de l'Orin Nano / NX |
| 7.x (Thor) | 10.x | CUDA 13 |

Un engine ne se transporte ni d'une version de TensorRT à une autre, ni d'un GPU à un autre.
On copie donc l'**ONNX** (`best_1280.onnx` pour R2) et on construit l'engine sur le Jetson. Il
y a deux voies.

### A. Ultralytics sur le Jetson

Cette voie suppose d'installer PyTorch et Ultralytics, en suivant le guide Jetson d'Ultralytics
pour les roues propres à chaque JetPack. On utilise les mêmes scripts que sur le PC :

```bash
python scripts/export.py --weights best.pt --imgsz 1280 --formats engine --device 0 --check-n 100   # FP16
python scripts/bench.py --weights best.pt best_1280_fp16.engine --imgsz 1280 --device 0 \
    --video data/demo/dut_track__video19.mp4 --name jetson_r2_1280
python scripts/demo.py --weights best_1280_fp16.engine --imgsz 1280 --source <vidéo|caméra>
```

L'engine produit garde les métadonnées Ultralytics (noms de classes, imgsz). `demo.py` le
charge donc directement. Si 1280 est trop lent sur le module, R1 à 640 est l'alternative
légère : mAP50 comparable, mais beaucoup plus de confusions avec les oiseaux (voir
`RESULTS.md`).

### B. `trtexec`, sans PyTorch

C'est la voie pour DeepStream ou un runtime C++.

```bash
/usr/src/tensorrt/bin/trtexec --onnx=best_1280.onnx --saveEngine=best_1280_fp16.plan --fp16
# ONNX dynamique : préciser les formes
/usr/src/tensorrt/bin/trtexec --onnx=best_1280_dyn.onnx --saveEngine=best_dyn_fp16.plan --fp16 \
    --minShapes=images:1x3x640x640 --optShapes=images:1x3x1280x1280 --maxShapes=images:1x3x1280x1280
```

`trtexec` affiche sa propre mesure de latence (« GPU Compute Time »), à comparer à la colonne
« Inférence ». Cet engine n'a pas l'en-tête de métadonnées Ultralytics : pour `demo.py`,
préférer la voie A. La sortie est le tenseur brut `1 × 5 × N` décrit plus haut ; la NMS est à
faire en aval, sauf export avec `--nms`.

### Mesurer sur Jetson

- Passer en mode de puissance maximal (`sudo nvpmodel -q` pour voir les modes, puis
  `sudo nvpmodel -m <MAXN>`), puis lancer `sudo jetson_clocks`.
- Surveiller `tegrastats` pendant la mesure.
- Toujours noter le module (Orin Nano, NX ou AGX), le JetPack et le mode de puissance.
  `bench.py` note le reste.

### FP16, INT8, DLA

- **FP16** : le réglage par défaut. Ici, l'écart avec PyTorch FP32 est négligeable (tableau de
  vérification ci-dessus). Le contrôler à nouveau sur le Jetson avec `export.py`.
- **INT8** (`--precision int8`, calibration sur `configs/drones.yaml`) : non testé. C'est plus
  rapide en principe, mais il faut **ré-évaluer la mAP** (`scripts/eval.py`). Les cibles de
  quelques pixels, notre cas difficile, sont les plus sensibles à la quantification, comme le
  montre déjà le doublon de l'engine R1 en FP16. Calibrer sur des images représentatives : ciel,
  petites cibles, contre-jour.
- **DLA** (Orin NX / AGX Orin, pas l'Orin Nano) : des accélérateurs dédiés en FP16 ou INT8
  (`device=dla:0` côté Ultralytics), avec repli GPU pour les couches non supportées. C'est une
  piste pour libérer le GPU.

### Piste : DeepStream

Les mesures locales montrent qu'avec TensorRT, le pré-traitement et le décodage pèsent plus
lourd que le réseau. Sur un CPU ARM, ce sera encore plus marqué. Une chaîne Jetson complète
utiliserait :

- le décodage vidéo matériel (NVDEC) et un pré-traitement sur GPU sans copie, au lieu de la
  letterbox numpy ;
- `nvinfer` avec l'ONNX ou l'engine et un parseur de boîtes YOLO ;
- `nvtracker` (NvDCF, IOU…) à la place de ByteTrack ;
- la logique de verrouillage (`scripts/lockon.py`) dans une *probe* sur les métadonnées.

## Limites

- Les mesures locales se font sur un **GPU de PC portable** (RTX 3070 Laptop). Elles ne
  remplacent pas une mesure sur Jetson.
- La machine est partagée. Les mesures retenues ont été faites dans une fenêtre vérifiée, sans
  autre calcul ; un navigateur restait ouvert.
  - Avec les runtimes compilés (ORT CUDA, TensorRT), le p90 de l'inférence reste à moins de
    11 % de la médiane : la mesure est stable.
  - PyTorch CUDA, limité par le CPU, est plus dispersé : jusqu'à +35 % pour R1 à 640.
  - Les chiffres à ±10 % près dépendent de l'état de la machine.
- La vidéo de mesure a été encodée à 30 fps depuis des JPEG (`DATA.md`). Le décodage d'un flux
  caméra réel aura un coût différent.
- Les exports à taille fixe ne tournent qu'à leur imgsz.
- Les engines locaux ne sont pas distribuables : ils sont liés à la RTX 3070 et à TensorRT 10.16.
  Seuls le `.pt` et l'ONNX vont dans la Release.
- Pas de code de pilotage : la chaîne s'arrête à la détection, au suivi et à l'erreur de
  visée.
