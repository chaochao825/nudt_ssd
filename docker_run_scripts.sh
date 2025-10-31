#!/bin/bash

# Docker run scripts for SSD detector testing
# These scripts demonstrate how to run the container with different configurations

IMAGE_NAME="nudt_ssd:latest"
INPUT_PATH="/path/to/input"
OUTPUT_PATH="/path/to/output"

echo "========================================"
echo "SSD Docker Run Scripts"
echo "========================================"

# 1. Adversarial Sample Generation
echo ""
echo "1. Adversarial Sample Generation (ADV)"
echo "----------------------------------------"
cat << 'EOF'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=ssd300 \
  -e BACKBONE=vgg16 \
  -e DATA=coco \
  -e CLASS_NUMBER=80 \
  -e ATTACK_METHOD=fgsm \
  -e EPSILON=0.031 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOF

# 2. Attack Evaluation
echo ""
echo "2. Attack Evaluation (ATTACK)"
echo "----------------------------------------"
cat << 'EOF'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=attack \
  -e MODEL=ssd300 \
  -e BACKBONE=vgg16 \
  -e DATA=coco \
  -e CLASS_NUMBER=80 \
  -e ATTACK_METHOD=pgd \
  -e EPSILON=0.031 \
  -e STEP_SIZE=0.008 \
  -e MAX_ITERATIONS=10 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOF

# 3. Defense Application
echo ""
echo "3. Defense Application (DEFEND)"
echo "----------------------------------------"
cat << 'EOF'
docker run --rm \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=ssd300 \
  -e BACKBONE=vgg16 \
  -e DATA=coco \
  -e CLASS_NUMBER=80 \
  -e DEFEND_METHOD=scale \
  ${IMAGE_NAME}
EOF

# 4. Model Training
echo ""
echo "4. Model Training (TRAIN)"
echo "----------------------------------------"
cat << 'EOF'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=train \
  -e MODEL=ssd300 \
  -e BACKBONE=vgg16 \
  -e DATA=coco \
  -e CLASS_NUMBER=80 \
  -e EPOCHS=100 \
  -e BATCH=8 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOF

# 5. BIM Attack
echo ""
echo "5. BIM Attack"
echo "----------------------------------------"
cat << 'EOF'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=ssd300 \
  -e ATTACK_METHOD=bim \
  -e EPSILON=0.031 \
  -e STEP_SIZE=0.008 \
  -e MAX_ITERATIONS=20 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOF

# 6. C&W Attack
echo ""
echo "6. C&W Attack"
echo "----------------------------------------"
cat << 'EOF'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=ssd300 \
  -e ATTACK_METHOD=cw \
  -e MAX_ITERATIONS=50 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOF

# 7. DeepFool Attack
echo ""
echo "7. DeepFool Attack"
echo "----------------------------------------"
cat << 'EOF'
docker run --rm --gpus all \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=ssd300 \
  -e ATTACK_METHOD=deepfool \
  -e MAX_ITERATIONS=50 \
  -e DEVICE=0 \
  ${IMAGE_NAME}
EOF

# 8. Compression Defense
echo ""
echo "8. Compression Defense"
echo "----------------------------------------"
cat << 'EOF'
docker run --rm \
  -v ${INPUT_PATH}:/project/input:ro \
  -v ${OUTPUT_PATH}:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=ssd300 \
  -e DEFEND_METHOD=comp \
  ${IMAGE_NAME}
EOF

echo ""
echo "========================================"
echo "Note: Replace INPUT_PATH and OUTPUT_PATH with actual paths"
echo "========================================"

