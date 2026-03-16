import json
import re
from pathlib import Path
from typing import Dict


# ----------------------------
# Static configuration
# ----------------------------

SPLIT = "val"  # or "train" or "test"

# Dataset layout: ROOT_DATA_DIR / <action_name> / <video_file>
ROOT_DATA_DIR = Path(f"/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/{SPLIT}_root")

# Output JSON path
OUTPUT_JSON_PATH = Path(f"/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/qwen_video_{SPLIT}_data.json")

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Conversation templates (Qwen-VL video finetune format)
TRAIN_HUMAN_PROMPT = "<video>\nWhat action is performed by the first responder in this video and why is it performed?"
TEST_HUMAN_PROMPT = "<video>\nWhat action is performed by the first responder in this video and why is it performed?"
# TEST_HUMAN_PROMPT = "<video>\nWhat action is performed by the first responder in this video and why is it performed?? Select the most approriate label from the following options. Only return the label, do not include any additional text."

ACTION_DESCRIPTION_JSON_PATH = Path("/standard/UVA-DSA/Keshara/EgoVLM/data_utils/actions_with_nl_descriptions_Qwen3-Coder-30B-A3B-Instruct.json")

def load_action_descriptions(json_path: Path) -> Dict[str, str]:
    # sample json structure:
#     {
#   "keysteps_descriptions": {
#     "approach_patient": "The responder is approaching a patient to initiate assessment and care while ensuring scene safety and spinal precautions. This involves systematically evaluating the patient's condition, determining priority for transport, and preparing for potential immobilization or advanced airway management as needed.",
#     "check_responsiveness": "The responder is assessing the patient's level of consciousness by gently tapping the patient's shoulder and shouting \"Are you okay?\" to determine if the patient responds appropriately. This action helps establish whether the patient is alert and oriented or if immediate intervention is required to maintain airway patency and circulation.",
#     "check_pulse": "The responder is assessing the patient's pulse manually to determine the presence of circulation, typically by palpating the carotid or radial artery for a sustained beat. This action guides further resuscitative interventions such as chest compressions or defibrillation, especially during cardiac arrest scenarios where pulse checks are critical for determining treatment course.",
    
    if not json_path.exists():
        raise FileNotFoundError(f"Action description JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        data = data.get("keysteps_descriptions", {})

        print(f"Loaded {len(data)} action descriptions from {json_path}")
    return data


def normalize_action_text(action_name: str, action_descriptions: Dict[str, str]) -> str:
    """Convert folder-style action names into a readable label."""
    # text = re.sub(r"[_\-]+", " ", action_name).strip()

    if SPLIT == "train":
        # For training, we want to use the natural language descriptions to encourage learning the mapping from video to text.
        return action_descriptions.get(action_name, action_name)
    else:
        # For val/test, we want to use the original action names as labels to evaluate whether the model can predict the correct label.
        return action_name



def build_video_dataset(root_dir: Path) -> list[dict]:
    dataset = []
    used_ids: set[str] = set()

    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DATA_DIR does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"ROOT_DATA_DIR is not a directory: {root_dir}")

    action_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])

    action_descriptions = load_action_descriptions(ACTION_DESCRIPTION_JSON_PATH)

    actions_list = list(action_descriptions.keys())

    for action_dir in action_dirs:
        action_name = action_dir.name
        action_text = normalize_action_text(action_name, action_descriptions)

        videos = sorted(
            [p for p in action_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
        )

        for video_path in videos:
            base_id = f"{action_name}_{video_path.stem}"
            sample_id = base_id
            counter = 1
            while sample_id in used_ids:
                counter += 1
                sample_id = f"{base_id}_{counter}"
            used_ids.add(sample_id)


            if SPLIT == "train":
                dataset.append(
                    {
                        "id": sample_id,
                        "video": str(video_path.resolve()),
                        "conversations": [
                            {
                                "from": "human",
                                "value": TRAIN_HUMAN_PROMPT,
                            },
                            {
                                "from": "gpt",
                                "value": action_text,
                            },
                        ],
                    }
                )
            else:
                dataset.append(
                    {
                        "id": sample_id,
                        "video": str(video_path.resolve()),
                        "conversations": [
                            {
                                "from": "human",
                                "value": TEST_HUMAN_PROMPT,
                            },
                            {
                                "from": "gpt",
                                "value": action_text,
                            },
                        ],
                    }
                )

    return dataset


def main() -> None:
    dataset = build_video_dataset(ROOT_DATA_DIR)
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Scanned root directory: {ROOT_DATA_DIR}")
    print(f"Generated samples: {len(dataset)}")
    print(f"Output JSON: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
