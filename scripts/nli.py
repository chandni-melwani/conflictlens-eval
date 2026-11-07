import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# Ensure CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_tokenizer = None
_model = None
_id2label = None


def get_nli_components():
    global _tokenizer, _model, _id2label
    if _model is None or _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        _model.to("cpu")

        raw_id2label = _model.config.id2label
        if not raw_id2label:
            raise ValueError(f"Model config for {MODEL_NAME} does not contain id2label mapping.")

        _id2label = {int(k): str(v).lower() for k, v in raw_id2label.items()}
        expected_labels = {"contradiction", "entailment", "neutral"}
        actual_labels = set(_id2label.values())
        if actual_labels != expected_labels:
            raise ValueError(
                f"Expected label set {expected_labels}, but got {actual_labels} from id2label: {_id2label}"
            )
    return _tokenizer, _model, _id2label


def nli_probs(premise: str, hypothesis: str) -> dict[str, float]:
    tokenizer, model, id2label = get_nli_components()
    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = F.softmax(logits, dim=-1)

    return {id2label[i]: float(probs[i].item()) for i in range(len(probs))}


def contradiction_score(a: str, b: str) -> float:
    probs_ab = nli_probs(a, b)
    probs_ba = nli_probs(b, a)
    return max(probs_ab["contradiction"], probs_ba["contradiction"])

