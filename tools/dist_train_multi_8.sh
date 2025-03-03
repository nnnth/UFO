# export JAVA_HOME=/home/tanghao/jdk1.8.0_152
# export JRE_HOME=${JAVA_HOME}/jre
# export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
# export PATH=${JAVA_HOME}/bin:$PATH

CONFIG=$1
GPUS=$2
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29505}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
echo MASTER_ADDR:$MASTER_ADDR
echo MASTER_PORT:$MASTER_PORT
echo RANK:$RANK
echo WORD_SIZE:$WORLD_SIZE
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=8 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
