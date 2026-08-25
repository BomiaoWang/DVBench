# DVBench data

This directory contains the released DVBench annotations and source-video metadata. License terms are in [`../DATA_LICENSE.md`](../DATA_LICENSE.md).

## Files

| File | Contents | Records |
|---|---|---:|
| `DVBench_QA.jsonl` | One benchmark QA object per line | 1,000 |
| `video_url.csv` | Video identifiers, titles, and original source URLs | 300 |

The QA release references all 300 distinct videos, and every referenced `video` has a matching `video_id`.

## QA schema

Each object in `DVBench_QA.jsonl` contains:

- `question_id`: unique question identifier.
- `question_type`: `EM`, `MCQ_single`, `MCQ_multiple`, or `Open_ended`.
- `video`: identifier joining to `video_url.csv.video_id`.
- `dimension`: `Narrative`, `Animation`, `Chart Perception`, `Chart Reasoning`, or `Alignment`.
- `question`: question text or subtitle-cloze context.
- `answer`: reference answer text.
- `distractor1`, `distractor2`, `distractor3`: distractors where applicable.
- `chart_type`: chart category where applicable.
- `animation_editorial_layer`: Animation subcategory where applicable.
- `chart_reas_type`: Chart Reasoning subcategory where applicable.
- `alignment_semantic_label`: `Data Insight` or `Data Context` where applicable.

`video_url.csv` contains `video_id`, `video_name`, and `url`.

## Loading

```python
import pandas as pd

qa = pd.read_json("data/DVBench_QA.jsonl", lines=True)
videos = pd.read_csv("data/video_url.csv", dtype={"video_id": str})
```

The lowercase public package reads this canonical JSONL directly:

```python
from dvbench import load_jsonl
records = load_jsonl("data/DVBench_QA.jsonl")
```

The canonical inference runner reads this JSONL directly and writes evaluator-ready JSONL predictions.

## Videos

Video files are not distributed in this Git repository. If you lawfully obtain them, the inference runner expects `data/videos/<video_id>.mp4` (unpadded and zero-padded numeric names are both resolved). URLs are references to third-party sources, not a grant of rights or a guarantee of continued availability. Follow each platform's terms and applicable copyright law.

## Important MCQ note

The release stores answer and distractor text rather than fixed A–D labels. `dvbench.build_prompt` shuffles choices deterministically per question and seed and returns both `label_to_option` and `correct_labels`. Preserve the prompt seed with every run; the evaluator defaults to seed `0`.
