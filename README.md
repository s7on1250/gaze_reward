# Gaze Reward RLHF

A reinforcement learning from human feedback (RLHF) package leveraging gaze-based reward models.

---

## 🚀 Installation

Installation and usage instructions are primarily the same as in the [official Gaze Reward repository](https://github.com/telefonica-scientific-research/gaze_reward).

---

## ⚙️ Usage

The main training script is `rlhf_rw/main.py`. Below are example commands for different gaze-based reward configurations.

### Examples of Prompts

| Model Configuration                            | Command    |
| ---------------------------------------------- | ---------- |
| **Roberta-based gaze model**                   | \`\`\`bash |
| python rlhf\_rw/main.py \\                     |            |
| -d nvidia/HelpSteer2 \\                        |            |
| -m meta-llama/Meta-Llama-3-8B \\               |            |
| --concat True \\                               |            |
| --use\_softprompt True \\                      |            |
| --learning\_rate 5e-5 \\                       |            |
| --logging\_steps 6000 \\                       |            |
| --lr\_scheduler\_type cosine\_with\_min\_lr \\ |            |
| --min\_lr\_ratio 0.7 \\                        |            |
| --batch\_size 1 \\                             |            |
| --fp\_dropout 0.1,0.3 \\                       |            |
| -fmv 2 \\                                      |            |
| --features\_used 0,1,0,1,0 \\                  |            |
| --seed 44                                      |            |

````|
| **Roberta on perturbed text**                 |```bash
python rlhf_rw/main.py \
  -d nvidia/HelpSteer2 \
  -m meta-llama/Meta-Llama-3-8B \
  --concat True \
  --use_softprompt True \
  --learning_rate 5e-5 \
  --logging_steps 6000 \
  --lr_scheduler_type cosine_with_min_lr \
  --min_lr_ratio 0.7 \
  --batch_size 1 \
  --fp_dropout 0.1,0.3 \
  -fmv 5 \
  --features_used 0,1,0,1,0 \
  --seed 44
```|
| **Roberta-large ensemble**                    |```bash
python rlhf_rw/main.py \
  -d nvidia/HelpSteer2 \
  -m mistralai/Mistral-7B-v0.1 \
  --concat True \
  --use_softprompt True \
  --learning_rate 5e-5 \
  --logging_steps 6000 \
  --lr_scheduler_type cosine_with_min_lr \
  --min_lr_ratio 0.7 \
  --batch_size 1 \
  --fp_dropout 0.1,0.3 \
  -fmv 6 \
  --features_used 0,1,0,1,0 \
  --seed 44 \
  --roberta_model_paths "PATHS" \
  --num_roberta_models 5
```|

---

## 📖 Customization

- **`-d` / `--dataset`**: Hugging Face dataset identifier (e.g., `nvidia/HelpSteer2`).
- **`-m` / `--model`**: Base language model (e.g., `meta-llama/Meta-Llama-3-8B`).
- **`--concat`**: Whether to concatenate gaze features (`True`/`False`).
- **`--use_softprompt`**: Use soft-prompt tuning (`True`/`False`).
- **`--learning_rate`**: Initial learning rate.
- **`--logging_steps`**: Steps between logging.
- **`--lr_scheduler_type`**: Scheduler type (e.g., `cosine_with_min_lr`).
- **`--min_lr_ratio`**: Minimum LR ratio for scheduler.
- **`--batch_size`**: Batch size per GPU.
- **`--fp_dropout`**: Dropout rates for attention and hidden layers.
- **`-fmv`**: Gaze model type.
- **`--features_used`**: Comma-separated gaze feature flags.
- **`--seed`**: Random seed for reproducibility.
- **`--roberta_model_paths`**: Comma-separated paths to pretrained RoBERTa models (for ensembles).
- **`--num_roberta_models`**: Number of RoBERTa models in ensemble.

---
