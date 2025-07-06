# 设置可见的 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1

# 自动计算 GPU 数量
NUM_TRAINERS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
NUM_TRAINERS=$((NUM_TRAINERS + 1))

# 启动 DDP 训练
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_TRAINERS train.py