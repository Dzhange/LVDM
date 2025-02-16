PROJ_ROOT="results/"                      # root directory for saving experiment logs
# EXPNAME="pusht_short_cond"          # experiment name 
DATADIR="datasets/push_t"    # dataset directory
CKPT_PATH="/home/ubuntu/Robot/LVDM/results/pusht_short_cond/checkpoints/last_summoning.ckpt"    # pretrained video autoencoder checkpoint
AEPATH="/home/ubuntu/Robot/LVDM/results/pusht_ae_64/checkpoints/epoch=0004-step=056399.ckpt"    # pretrained video autoencoder checkpoint

CONFIG="configs/pusht/short_cond.yaml"

export TOKENIZERS_PARALLELISM=false
python scripts/cond_sample_exp.py \
    --ckpt_path $CKPT_PATH \
    --first_stage_ckpt $AEPATH \
    --config_path $CONFIG \
    --save_dir results/sample_cond \
    --n_samples 4 \
    --num_frames 16 \
    --sample_type ddim \
    --ddim_steps 50