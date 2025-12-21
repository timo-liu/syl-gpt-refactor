#!/usr/bin/env bash

TORCHRUN="/mnt/data/syl-gpt-refactor/Training/.venv/bin/torchrun"
TRAIN_MODULE="Training.train"

CONFIG_BASE="/mnt/data/syl-gpt-refactor/Configs"
WEIGHTS_BASE="/mnt/data/weights/old_goods"

LANGS=("eng" "span")
PARADIGMS=("syl" "uni" "bpe")
CV_COUNTERS=({0..9})

# Arbitrary task names (NOT tied to paradigms)
TASKS=(
  "syllables"
  "word"
)

# Arbitrary task prefixes ("" = no prefix)
TASK_PREFIXES=(
  ""
  # "exp1."
  # "abl."
  # "foo."
)

# Language → data directory
declare -A DATA_DIRS=(
  [eng]="/mnt/data/syl-gpt-refactor/Data/english_data/bins"
  [span]="/mnt/data/syl-gpt-refactor/Data/spanish_data/bins"
)

# (Optional) pretrained weights per (lang, paradigm)
declare -A WEIGHTS_FILES=(
  [eng_syl]="$WEIGHTS_BASE/eng_syl_good.pth"
  [eng_uni]="$WEIGHTS_BASE/eng_uni_good.pth"
  [eng_bpe]="$WEIGHTS_BASE/eng_bpe_good.pth"
  [span_syl]="$WEIGHTS_BASE/span_syl_good.pth"
  [span_uni]="$WEIGHTS_BASE/span_uni_good.pth"
  [span_bpe]="$WEIGHTS_BASE/span_bpe_good.pth"
)

for lang in "${LANGS[@]}"; do
  DATA_DIR="${DATA_DIRS[$lang]}"

  for paradigm in "${PARADIGMS[@]}"; do
    CONFIG="trained_${lang}_${paradigm}_config.json"

    KEY="${lang}_${paradigm}"
    WEIGHTS_PATH="${WEIGHTS_FILES[$KEY]}"

    for prefix in "${TASK_PREFIXES[@]}"; do
      for base_task in "${TASKS[@]}"; do
        TASK="${prefix}${base_task}"

        for cv in "${CV_COUNTERS[@]}"; do
          WEIGHTS_ARGS=()
          if [[ -n "$WEIGHTS_PATH" && -f "$WEIGHTS_PATH" ]]; then
            WEIGHTS_ARGS=(--weights_path "$WEIGHTS_PATH")
            echo "→ Using pretrained weights: $WEIGHTS_PATH"
          else
            echo "→ Training from scratch: ${KEY}, fold ${cv}"
          fi

          "$TORCHRUN" --nproc_per_node=2 \
            -m "$TRAIN_MODULE" \
            "$CONFIG_BASE/$CONFIG" \
            "$DATA_DIR" \
            "/mnt/data/weights/finetuned" \
            "${WEIGHTS_ARGS[@]}" \
            --task "$TASK" \
            --cross_val_counter "$cv"
        done
      done
    done
  done
done
