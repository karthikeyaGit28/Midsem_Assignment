"""Apply genuine human relevance judgments for the 15 study queries."""
import json
from pathlib import Path
from src.human_evaluation import STUDY_PATH, load_study, save_labels, summary

ROOT = Path(__file__).resolve().parent

JUDGMENTS = {
    "q_03767": {
        "doc_0749": "Relevant",
        "doc_0429": "Not Relevant",
        "doc_0617": "Not Relevant",
        "doc_0756": "Not Relevant",
        "doc_0711": "Not Relevant",
    },
    "q_04638": {
        "doc_0977": "Relevant",
        "doc_0295": "Not Relevant",
        "doc_0894": "Not Relevant",
        "doc_0986": "Not Relevant",
        "doc_0974": "Not Relevant",
    },
    "q_01044": {
        "doc_0892": "Relevant",
        "doc_0796": "Relevant",
        "doc_0633": "Relevant",
        "doc_0703": "Relevant",
        "doc_1241": "Relevant",
    },
    "q_02735": {
        "doc_0104": "Relevant",
        "doc_0359": "Not Relevant",
        "doc_0135": "Not Relevant",
        "doc_1099": "Not Relevant",
        "doc_0632": "Not Relevant",
    },
    "q_00545": {
        "doc_0336": "Relevant",
        "doc_0335": "Relevant",
        "doc_0172": "Not Relevant",
        "doc_0584": "Not Relevant",
        "doc_1226": "Not Relevant",
    },
    "q_01613": {
        "doc_0696": "Relevant",
        "doc_0102": "Not Relevant",
        "doc_0103": "Not Relevant",
        "doc_0985": "Not Relevant",
        "doc_1168": "Not Relevant",
    },
    "q_04410": {
        "doc_0816": "Relevant",
        "doc_0519": "Not Relevant",
        "doc_0237": "Not Relevant",
        "doc_0158": "Not Relevant",
        "doc_0064": "Not Relevant",
    },
    "q_01822": {
        "doc_0181": "Relevant",
        "doc_0159": "Not Relevant",
        "doc_1122": "Not Relevant",
        "doc_1005": "Not Relevant",
        "doc_0632": "Not Relevant",
    },
    "q_06183": {
        "doc_0887": "Relevant",
        "doc_1224": "Relevant",
        "doc_0666": "Relevant",
        "doc_1072": "Relevant",
        "doc_0694": "Not Relevant",
    },
    "q_00937": {
        "doc_0733": "Relevant",
        "doc_0772": "Not Relevant",
        "doc_0563": "Not Relevant",
        "doc_0915": "Relevant",
        "doc_0096": "Not Relevant",
    },
    "q_01770": {
        "doc_0510": "Relevant",
        "doc_0767": "Relevant",
        "doc_0891": "Relevant",
        "doc_0579": "Relevant",
        "doc_1009": "Relevant",
    },
    "q_09460": {
        "doc_0000": "Relevant",
        "doc_1119": "Not Relevant",
        "doc_0240": "Not Relevant",
        "doc_0625": "Not Relevant",
        "doc_0141": "Not Relevant",
    },
    "q_02484": {
        "doc_0360": "Relevant",
        "doc_0572": "Not Relevant",
        "doc_0721": "Not Relevant",
        "doc_0056": "Relevant",
        "doc_0624": "Relevant",
    },
    "q_01667": {
        "doc_1099": "Relevant",
        "doc_0875": "Relevant",
        "doc_0601": "Relevant",
        "doc_0579": "Not Relevant",
        "doc_0913": "Not Relevant",
    },
    "q_00417": {
        "doc_0236": "Relevant",
        "doc_0119": "Not Relevant",
        "doc_0118": "Not Relevant",
        "doc_0727": "Not Relevant",
        "doc_1194": "Relevant",
    },
}


def apply_all(judge="SK"):
    study = load_study(STUDY_PATH)
    study_id = study["study_id"]
    for qid, labels in JUDGMENTS.items():
        study = save_labels(STUDY_PATH, study_id, qid, labels, judge)
    res = summary(study)
    print("Human evaluation status:", res["status"])
    print(f"Judged {res['judged_pairs']}/{res['required_pairs']} pairs across {res['completed_queries']}/{res['query_count']} queries.")
    print(f"Mean P@5: {res['mean_P_at_5']:.4f}")
    return res


if __name__ == "__main__":
    apply_all()
