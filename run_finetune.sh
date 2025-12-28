#!/usr/bin/env bash

TORCHRUN="/mnt/data/syl-gpt-refactor/Training/.venv/bin/torchrun"
PYTHON="/mnt/data/syl-gpt-refactor/Training/.venv/bin/python3"

TRAIN_MODULE="Training.train"
EVAL_MODULE="Training.eval_finetune"

CONFIG_BASE="/mnt/data/syl-gpt-refactor/Configs"
WEIGHTS_BASE="/mnt/data/weights/old_goods"
FINETUNED_BASE="/mnt/data/weights/finetuned"

LANGS=("eng" "span")
PARADIGMS=("syl" "uni" "bpe")
CV_COUNTERS=({0..9})

TASKS=(
	# "syllables"
	# "word"
	"g2p"
)

TASK_PREFIXES=(
	""
)

declare -A DATA_DIRS=(
	[eng]="/mnt/data/syl-gpt-refactor/Data/english_data/bins"
	[span]="/mnt/data/syl-gpt-refactor/Data/spanish_data/bins"
)

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
          WEIGHT_FILE="${FINETUNED_BASE}/${TASK}_${lang}_${paradigm}_${cv}_finetuned.pth"

          # ---- SKIP TRAIN IF EXISTS ----
          if [[ -f "$WEIGHT_FILE" ]]; then
            echo "✓ Finetuned weights already exist: $WEIGHT_FILE"
          else
            WEIGHTS_ARGS=()
            if [[ -n "$WEIGHTS_PATH" && -f "$WEIGHTS_PATH" ]]; then
              WEIGHTS_ARGS=(--weights_path "$WEIGHTS_PATH")
              echo "→ Using pretrained weights: $WEIGHTS_PATH"
            else
              echo "→ Training from scratch: ${KEY}, fold ${cv}"
            fi

            # ---- TRAIN ----
            "$TORCHRUN" --nproc_per_node=2 \
              -m "$TRAIN_MODULE" \
              "$CONFIG_BASE/$CONFIG" \
              "$DATA_DIR" \
              "$FINETUNED_BASE" \
              "${WEIGHTS_ARGS[@]}" \
              --task "$TASK" \
              --cross_val_counter "$cv"

            if [[ $? -ne 0 ]]; then
              echo "✗ Training failed — skipping eval"
              continue
            fi
          fi

          # ---- EVAL ----
          if [[ ! -f "$WEIGHT_FILE" ]]; then
            echo "✗ Missing finetuned weights: $WEIGHT_FILE"
            continue
          fi

          echo "→ Evaluating: $WEIGHT_FILE"

          "$PYTHON" -m "$EVAL_MODULE" \
            "$WEIGHT_FILE" \
            "$lang" \
            "$paradigm" \
            "$cv" \
            "$TASK" \
            "$DATA_DIR"
        done
			done
		done
	done
done
