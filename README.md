# brainmets
Segmentation for brain mets.  Currently only takes in bravo.

## Docker Image
From `brainmets/docker`, run `docker-compose up --build`

For mounting the volumes, the only necessary line is the first one, e.g.:
`/home/darvin/nas/brainmets/:/home/bmcv/brainmets`

Please change the host volume to wherever you saved/cloned this project directory.
The only reason the image directory has to be /home/bmcv/brainmets is because
in the Dockerfile, I passed that directory to the PYTHONPATH.  If you replicate
that process, you can mount the project directory to wherever you want.  You 
can even COPY the directory in the Dockerfile as well.  This code is optimized
for running experiments and debugging, not for inference.

The rest don't matter as long as you give the pertinent files.

## Preprocessing
Run: `python brainmets/preprocessing/preprocess_for_training.py --pData path/to/data --pSave path/to/save`.

Currently, this only saves the BRAVO files.

### Pipeline
Once the volume has been loaded, we `brainmets.utils.io.process_volume`.  This does the following:
1. unit normalizes the volume (`brainmets.utils.io.normalize_volume`)
2. binarizes the MR using otsu's method just to find roughly the skull
3. finds the "bounding box" that contains the head
4. crops out the head and resizes the volume to be 512x512xnum_frames
5. we save the original coordinates of the "bounding box"

## Training
COMING.  CURRENTLY TRAINING IS DONE QUICKLY IN `notebooks/toy_training.ipynb`.

## Configuration Files
COMING.

## Model
Download the model from this google drive link: https://drive.google.com/file/d/10P8rPojbBBKk6aGqI_7qWk-W9SdUPpwu/view?usp=sharing

## Inference
Run `python /home/bmcv/brainmets/tools/run_inference_nii.py --pModel path/to/model.pth --pNii path/to/.nii`

Currently, the inference file takes in the path to the model and a path to the niifty file.  It will load in both, process the niifty volume, and then save the output.

The output will be saved in the same directly with `.out.nii.gz` appended to the originial niifty file's name.

Use this inference file as an example of how to run the model.