## Data preparation
Step 1: prepara free-form surfaces that will be used in the training process.

Step 2: use the method described in sub-section "2.1 Dataset preparation" to generate corresponding training data.

Step 3: put the curvature and height cloud maps under directory './datasets/GridData/train_A' and './datasets/GridData/train_H', respectively, and then place the free-form grid structure images under folder './datasets/GridData/train_B'.

## Train process
use the command line below to start the training under the folder GridGAN, namely the foder where train.py file exists:
```bash
!# all parameters, except the parameters below, will be set as default and are stored under folder './options'  
python train.py --name GridGAN --tf_log --label_nc 0 --lambda_feat 15 --no_flip --dataroot ./datasets/GridData --n_downsample_global 4 --n_blocks_global 9 --lr 0.00005 --niter 50 --niter_decay 50
```

## Test process
After the train of the model has finished, run the command below to check the test result.
```bash
python test.py --name GridGAN --dataroot ./datasets/GridData --n_downsample_global 4 --n_blocks_global 9 --which_epoch latest
```
