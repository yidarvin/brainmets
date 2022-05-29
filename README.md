# brainmets
Segmentation for brain mets.

## Docker Image
From `brainmets/docker`, run `docker-compose up --build`

## Preprocessing
Run: `python brainmets/preprocessing/preprocess_for_training.py --pData data_nas/StanfordMets/ data_scratch/StanfordMets512 --bBravo 1 --bFlair 1 --bT1_gd 1 --bT1_pre 1 --bSeg 1`.