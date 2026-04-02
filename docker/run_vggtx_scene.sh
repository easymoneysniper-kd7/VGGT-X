#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${IMAGE_NAME:=vggt-x:cuda121}"
: "${SCENE_DIR:=${1:-/home/cw/Desktop/project/zyc/data}}"
: "${GPU_DEVICES:=0}"
: "${POST_FIX:=_vggt_x}"
: "${CHUNK_SIZE:=128}"
: "${MAX_QUERY_PTS:=2048}"
: "${SAVE_DEPTH:=0}"
: "${TOTAL_FRAME_NUM:=}"
: "${EXTRA_ARGS:=}"
: "${MOUNT_ROOT:=$(dirname "$(dirname "${SCENE_DIR}")")}"
: "${CACHE_DIR:=${PROJECT_ROOT}/.docker-cache}"
: "${OUTPUT_SCENE_NAME:=$(basename "$(dirname "${SCENE_DIR}")")_$(basename "${SCENE_DIR}")}"
: "${OUTPUT_DIR:=${PROJECT_ROOT}/output/${OUTPUT_SCENE_NAME}}"
: "${RUN_AS_HOST_USER:=0}"
: "${HOST_UID:=$(id -u)}"
: "${HOST_GID:=$(id -g)}"

SOURCE_RESULT_DIR="$(dirname "${SCENE_DIR}")${POST_FIX}/$(basename "${SCENE_DIR}")"

if [[ ! -d "${SCENE_DIR}/images" ]]; then
    echo "Expected image directory at ${SCENE_DIR}/images"
    exit 1
fi

mkdir -p "${CACHE_DIR}/torch" "${CACHE_DIR}/hf"

RUN_ARGS=(
    python demo_colmap.py
    --scene_dir "${SCENE_DIR}"
    --post_fix "${POST_FIX}"
    --chunk_size "${CHUNK_SIZE}"
    --max_query_pts "${MAX_QUERY_PTS}"
    --shared_camera
    --use_ga
)

DOCKER_USER_ARGS=()

if [[ "${RUN_AS_HOST_USER}" == "1" ]]; then
    DOCKER_USER_ARGS=(--user "${HOST_UID}:${HOST_GID}")
fi

if [[ "${SAVE_DEPTH}" == "1" ]]; then
    RUN_ARGS+=(--save_depth)
fi

if [[ -n "${TOTAL_FRAME_NUM}" ]]; then
    RUN_ARGS+=(--total_frame_num "${TOTAL_FRAME_NUM}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
    RUN_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

docker run --rm \
    --name "vggt-x-$(date +%Y%m%d-%H%M%S)" \
    --gpus "device=${GPU_DEVICES}" \
    --ipc=host \
    --shm-size=64g \
    "${DOCKER_USER_ARGS[@]}" \
    -e CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
    -e TORCH_HOME=/opt/torch-cache \
    -e HF_HOME=/opt/hf-cache \
    -v "${PROJECT_ROOT}:/workspace/VGGT-X" \
    -v "${MOUNT_ROOT}:${MOUNT_ROOT}" \
    -v "${CACHE_DIR}/torch:/opt/torch-cache" \
    -v "${CACHE_DIR}/hf:/opt/hf-cache" \
    -w /workspace/VGGT-X \
    "${IMAGE_NAME}" \
    "${RUN_ARGS[@]}"

mkdir -p "${OUTPUT_DIR}/sparse/0"
cp -a "${SOURCE_RESULT_DIR}/sparse/0/." "${OUTPUT_DIR}/sparse/0/"

if [[ -f "${SOURCE_RESULT_DIR}/matches.pt" ]]; then
    cp -a "${SOURCE_RESULT_DIR}/matches.pt" "${OUTPUT_DIR}/"
fi

if [[ -f "${SOURCE_RESULT_DIR}/loss_curve_pose_opt.png" ]]; then
    cp -a "${SOURCE_RESULT_DIR}/loss_curve_pose_opt.png" "${OUTPUT_DIR}/"
fi

if [[ ! -e "${OUTPUT_DIR}/images" ]]; then
    ln -s "${SCENE_DIR}/images" "${OUTPUT_DIR}/images"
fi

echo "Synced sparse output to ${OUTPUT_DIR}"
