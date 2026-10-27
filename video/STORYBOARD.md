# Vidéo de présentation — storyboard et fabrication

Livrables : [`docs/assets/presentation.mp4`](../docs/assets/presentation.mp4) (≤ 10 Mo, limite
d'upload vidéo GitHub) et l'affiche [`docs/assets/presentation_poster.jpg`](../docs/assets/presentation_poster.jpg).

- **Public** : visiteurs du dépôt GitHub, recruteurs techniques.
- **Format** : 1920×1080, 30 i/s, H.264 High `yuv420p`, `+faststart`, environ 114 s, **muette**.
  Tout passe par l'image et le texte (≥ 40 px), car les vidéos GitHub démarrent sans son.
- **Règle** : tout ce qui est montré est réel. Les images viennent des séquences de test et des
  données d'entraînement. Les chiffres sont lus dans les fichiers de résultats par
  [`facts.py`](facts.py) ; le tableau de traçabilité est en fin de page.

## Déroulé

| t (s) | Scène | Contenu | Matière |
|---|---|---|---|
| 0–8 | `hook` | Bandeau « Séquence de test, jamais vue à l'entraînement ». Vrai clip de test plein écran, avec l'overlay de verrouillage : réticule, `LOCK #id`, dx/dy, FPS. Le segment est choisi pour qu'une acquisition SEARCHING → LOCKED ait lieu vers 1,2 s. Titre « drone-detect » en bas à gauche. | DUT video18 (1080p, test) |
| 8–18 | `context` | Le problème (guidage final par vision) et 4 difficultés, chacune sur une vraie image : cibles minuscules (taille médiane du test), oiseaux et aéronefs, fonds encombrés, temps réel embarqué (33 ms par image à 30 i/s). | V_DRONE_035, V_BIRD_020, video19 |
| 18–28 | `pipeline` | Chaîne animée Caméra → YOLO11n → ByteTrack → Verrouillage → dx, dy, illustrée sur **une même vraie frame** : image brute, boîtes, traînée de suivi, overlay de la démo, valeurs dx/dy. Bloc en pointillés « Guidage — hors périmètre » et mention « Aucun code de contrôle de vol ni de charge utile ». | DUT video19 |
| 28–38 | `lock` | Machine à états SEARCHING / LOCKED / COASTING avec les réglages réels de `demo.py`. Convention dx, dy dessinée sur une vraie frame, formules, écriture CSV. | `scripts/demo.py`, `scripts/lockon.py`, video19 |
| 38–48 | `data` | Mosaïque de 9 vrais échantillons d'entraînement (vérité terrain en vert), sources et licences, barre train / val / test, découpage par séquence. | `data/yolo`, `data/yolo/stats.md` |
| 48–59 | `demo19` | Démo 1 : vidéo + overlay, **loupe** sur la cible, **graphe dx/dy synchronisé**. Le graphe trace l'entrée d'un contrôleur de guidage, avec une bande d'état sous la courbe. Une légende SEARCHING / LOCKED / COASTING allume l'état courant à chaque image. | DUT video19 (test) |
| 59–69 | `demo35` | Démo 2 : petite cible sous un ciel couvert, séquence entière. Elle montre toute la machine à états : attente, acquisition, perte (COASTING), relâche, ré-acquisition sous un nouvel ID. | Halmstad V_DRONE_035 (test) |
| 69–79 | `birds` | Oiseaux : R1 (640) et R2 (1280) côte à côte sur la même séquence, avec un compteur d'images verrouillées. Puis bilan sur tout le test, en barres : fausses détections sur les 55 images d'oiseaux, R1 contre R2. Les phrases sont tirées des pistes : R1 se verrouille sur des oiseaux, R2 sur aucune des 327 images. | Halmstad V_BIRD_020 (test) |
| 79–93 | `results` | 4 chiffres clés animés : rappel et précision à IoU 0,3, mAP50 (+ mAP50-95), FPS TensorRT de bout en bout. Puis SAHI contre image entière sur le 1080p DUT (mAP50 et ms/image). | `runs/eval`, `runs/sahi`, `runs/bench` |
| 93–107 | `limits` | Limites : boîtes approximatives, faux positifs fixes, contre-jour / nuit / IR non testés, pas de FPV réel. Les **deux faux positifs fixes de R2 sont montrés en vraies images**, tirés de la galerie d'échecs : branches en fleurs et élément de toit. Puis prochaines étapes : Jetson + TensorRT, filtre de mouvement, fusion audio + RF, guidage terminal. | `docs/RESULTS.md`, `docs/assets/failures/` |
| 107–114 | `end` | Coins de verrouillage qui se referment sur « drone-detect », URL du dépôt (à compléter), auteur, crédits des données et licences. | — |

Le minutage exact s'obtient avec `.venv/bin/python video/render.py --list`.

## Choix et points d'honnêteté

- **Overlay redessiné, pas réinventé.** [`track_dump.py`](track_dump.py) exécute la même boucle
  que `scripts/demo.py` : même modèle, `model.track` + ByteTrack, `LockOn`, réglages par défaut
  lus dans `demo.parse_args`. Il enregistre les pistes et l'état du verrou de chaque frame.
  [`clips.py`](clips.py) appelle ensuite **`demo.draw_overlay`** sur la frame agrandie. Les
  clips Halmstad, en 640×512, sont ainsi nets ; un overlay incrusté à 640 px puis agrandi serait
  flou. Le FPS affiché dans l'overlay est celui mesuré pendant l'enregistrement des pistes, sur
  le matériel indiqué en haut à droite de l'overlay.
- **Loupe et graphe** : la loupe recadre la frame brute autour de la cible ; seul son centre est
  lissé, pour un affichage stable. Les coins sont dessinés à la position réelle de la boîte.
  Le graphe trace les dx, dy du CSV, en trait plein en LOCKED et atténué en COASTING (dernière
  position, donnée périmée). La courbe est interrompue en SEARCHING.
- **Oiseaux** : la vidéo dure au plus 6,5 s, et elle serait coupée avant un éventuel verrou de
  R2. Les phrases à l'écran (qui se verrouille, sur quoi, sur toute la séquence) sont calculées
  depuis les pistes. Le compteur sous chaque vidéo indique « verrouillé : n / images vues », pour
  ne pas le confondre avec les 55 images d'oiseaux du test du bilan.
- **Fenêtres des démos, vérifiées à l'œil** (planches de contrôle des frames verrouillées) :
  - accroche, video18 frames 527 à 766 : le drone sort de derrière un arbre, puis il est
    réacquis sous #3 à 1,2 s. #1 et #3 sont bien le drone.
  - demo19, video19 frames 632 à 961 : verrou #1 continu. Après la frame 995, le verrou passe
    en COASTING : le drone descend sous la ligne du toit, près d'une lucarne. Viennent ensuite
    #10, #29 et #36. Les vérifications montrent que #10 et #36 portent bien sur le drone (frames
    1015, 1100, 1180, 1250). Ce passage n'est pas utilisé, mais il n'y a pas de faux verrou.
  - demo35, V_DRONE_035 entier : SEARCHING → LOCKED #2 → COASTING → relâche → LOCKED #15.
  - oiseaux, V_BIRD_020 frames 0 à 194 : R1 se verrouille sur de vrais oiseaux (#56, #109 ; et
    #235 plus loin dans la séquence). R2 ne se verrouille sur aucune des 327 images.
- **Faux positifs fixes** (scène « limites ») : images de la galerie d'échecs de l'évaluation
  de R2 (`.pt`, seuil F1-optimal). Avec l'engine et le verrou de la démo, R2 ne se verrouille
  pas sur ces branches. Un premier passage sur CPU avec le `.pt` et `lock-conf` 0,5 le faisait.
- **IoU 0,3** : présenté comme « la cible est trouvée, boîte approximative ». La justification
  vient de `docs/RESULTS.md` : la visée n'utilise que le centre de la boîte.
- **FPS** : mesurés sur RTX 3070 Laptop, ce qui est dit à l'écran (« pas encore sur Jetson »).
  Si `runs/bench/*.json` manque, la tuile affiche la latence PyTorch de `metrics.json`.
- **Valeurs calculées, non lues dans un fichier** : 33 ms (1/30 s) ; « ×3 » pour SAHI
  (57 ms / 19 ms, tous deux lus dans `runs/sahi/…/metrics.json`) ; « 15 à 40 px » et « moins de
  20 px », repris du texte de `docs/RESULTS.md`.

## Identité visuelle

- Fond sombre (`#090C11`) avec une grille très discrète, panneaux `#141921`.
- Couleurs d'état **identiques à l'overlay de la démo** : LOCKED rouge `#FF2828`, COASTING orange
  `#FF9600`, SEARCHING jaune `#FFDC00`, pistes non verrouillées cyan.
- Séries dx `#3987E5` et dy `#199E70`. La paire a été validée sur le fond sombre : séparation
  daltonisme ΔE 19,6, vision normale 20,9. Le jaune et l'orange restent réservés aux états.
- Polices libres (SIL OFL 1.1), téléchargées par [`fetch_fonts.py`](fetch_fonts.py) :
  **Inter** pour le texte, **JetBrains Mono** pour les valeurs « instrument » (chiffres, états,
  formules). Taille minimale 40 px. Les polices ne sont pas versionnées (`video/fonts/` est
  ignoré par git) : `fetch_fonts.py` les télécharge (empreintes SHA-256 vérifiées), et
  `render.py` le lance tout seul si elles manquent. Les lectures d'instrument (dx, dy) gardent le point décimal
  du CSV et de l'overlay ; le texte courant utilise la virgule.
- Transitions : fondus courts vers le fond (0,35 s) et entrées glissées avec easing.

## Fabriquer la vidéo

Prérequis : `.venv` du projet (Pillow, OpenCV, numpy, ultralytics), `ffmpeg` avec libx264, les
vidéos de démo `data/demo/*.mp4` (`scripts/data/make_video.py`) et les poids R1/R2 dans `runs/train/`.

```bash
export YOLO_CONFIG_DIR=$PWD/.ultralytics
.venv/bin/python video/fetch_fonts.py                       # polices (sinon lancé par render.py)
# Pistes de démo, mêmes réglages que les clips d'export (outputs/demo/make_clips.sh) ; GPU au repos :
# le FPS affiché dans l'overlay est celui mesuré pendant cette passe.
E=runs/train/r2_y11n_1280_ft/weights/best_1280_fp16.engine
.venv/bin/python video/track_dump.py --device 0 --weights $E --imgsz 1280 --conf 0.1 --lock-conf 0.4 --force
.venv/bin/python video/track_dump.py --device 0 --weights runs/train/r1_y11n_640/weights/best.pt --imgsz 640 \
    --conf 0.1 --lock-conf 0.4 --sources data/demo/halmstad__V_BIRD_020.mp4 --tag r1 --force
# Commande de rendu unique : scènes → master → MP4 ≤ 10 Mo + affiche + images de contrôle
.venv/bin/python video/render.py --all                      # ≈ 1 min 40 sur la machine locale
```

- `render.py --preview 2 6 --scenes demo19` : aperçus PNG dans `outputs/video/preview/`.
- `render.py --scenes hook` : re-rend une scène, puis ré-assemble et ré-encode le tout.
- Sorties intermédiaires (ignorées par git) dans `outputs/video/` : `tracks/`, `segments/`,
  `master.mp4` (quasi sans perte), `review/` (2 images par scène, extraites du MP4 final).
- Encodage final : x264 `veryslow` en 2 passes, avec un débit calculé pour viser 9,6 Mo
  (`--max-mb`). Si le fichier dépasse, l'encodage est relancé avec un débit réduit.

## Traçabilité des chiffres

`.venv/bin/python video/facts.py` réimprime ce tableau depuis les fichiers.

| Nom | Valeur | Source |
|---|---|---|
| `map50` | 0.7814 | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : ultralytics_val.mAP50` |
| `map5095` | 0.2648 | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : ultralytics_val.mAP50-95` |
| `p50` | 0.7779 | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : ultralytics_val.precision` |
| `r50` | 0.8118 | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : ultralytics_val.recall` |
| `p30` | 0.9053 | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : common_eval.pr_iou03.best_f1.precision` |
| `r30` | 0.9506 | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : common_eval.pr_iou03.best_f1.recall` |
| `n_test_img` | 2326 | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : common_eval.pr.n_images` |
| `n_test_gt` | 1982 | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : common_eval.pr.n_gt` |
| `lat_ms_val` | 9.84 | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : ultralytics_val.speed_ms_per_image_batched (pré + inf + post)` |
| `hw_eval` | NVIDIA GeForce RTX 3070 Laptop GPU (CUDA 13.0, torch 2.14.1+cu130) | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : hardware` |
| `dut_map50` | 0.8191 | `runs/eval/r2_y11n_1280_ft_1280_test/metrics.json : common_eval.by_source.dut_track.coco.mAP50` |
| `dut_sahi_map50` | 0.8849 | `runs/sahi/r2_y11n_1280_ft_1280_s640at640_test/metrics.json : modes.sahi.by_source.dut_track.coco.mAP50` |
| `dut_full_ms` | 18.87 | `runs/sahi/r2_y11n_1280_ft_1280_s640at640_test/metrics.json : modes.full.ms_per_image_by_source.dut_track` |
| `dut_sahi_ms` | 56.98 | `runs/sahi/r2_y11n_1280_ft_1280_s640at640_test/metrics.json : modes.sahi.ms_per_image_by_source.dut_track` |
| `r2_fp_bird` | 2 | `runs/failures/r2_y11n_1280_ft_1280_test/summary.json : by_category.fp_bird` |
| `r1_fp_bird` | 52 | `runs/failures/r1_draft/summary.json : by_category.fp_bird` |
| `n_bird_img` | 55 | `data/yolo/test/images : halmstad__V_BIRD_*` |
| `n_train` | 19495 | `data/yolo/stats.md : train, images` |
| `n_val` | 1824 | `data/yolo/stats.md : val, images` |
| `n_test` | 2326 | `data/yolo/stats.md : test, images` |
| `n_total` | 23645 | `data/yolo/stats.md : total, images` |
| `n_test_seq` | 49 | `data/yolo/stats.md : test, séquences` |
| `test_median_px` | 34.0 | `data/yolo/stats.md : test, médiane √(w·h)` |
| `trt_fps` | 391.1 | `runs/bench/final_gpu_r2_1280.json : fps_model (TensorRT FP16, 1280)` |
| `trt_fps_video` | 106.4 | `runs/bench/final_gpu_r2_1280.json : video.fps (lecture + inférence + suivi + verrou, 1080p)` |
| `trt_gpu` | GPU NVIDIA GeForce RTX 3070 Laptop GPU | `runs/bench/final_gpu_r2_1280.json : hardware` |

Réglages du verrou utilisés (lus dans les pistes, affichés dans la scène « verrouillage ») : `conf` 0.1, `lock-conf` 0.4, 3 images, COASTING 15 images, `bytetrack.yaml`.

Pistes enregistrées (`outputs/video/tracks/`, RTX 3070 Laptop, machine au repos) :

| Fichier | Poids | Frames | FPS pipeline moyen | LOCKED / COASTING / SEARCHING |
|---|---|---|---|---|
| `dut_track__video18__r2.json` | `best_1280_fp16.engine` @1280 | 1320 | 110.88 | 1258 / 15 / 47 |
| `dut_track__video19__r2.json` | `best_1280_fp16.engine` @1280 | 1300 | 110.36 | 1221 / 49 / 30 |
| `halmstad__V_BIRD_020__r1.json` | `best.pt` @640 | 327 | 121.71 | 86 / 34 / 207 |
| `halmstad__V_BIRD_020__r2.json` | `best_1280_fp16.engine` @1280 | 327 | 147.74 | 0 / 0 / 327 |
| `halmstad__V_DRONE_035__r2.json` | `best_1280_fp16.engine` @1280 | 302 | 126.23 | 237 / 15 / 50 |

