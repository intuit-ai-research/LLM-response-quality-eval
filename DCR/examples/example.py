import json
import os

import pandas as pd

from DCR.src.evaluator import evaluate
from DCR.src.improver import improve
from utils.llm import LLM_OpenAI

llm = LLM_OpenAI('gpt-4')
model_config = {"temperature": 0}
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = "qags_cnndm.json"
OUTPUT_DCE_AMC = "test_gags.csv"
OUTPUT_RAI = "test_gags_improved.csv"


def evalute_example():
    data = get_qags_data()
    print("data retreived")
    res = evaluate(llm, model_config, data, worker_count=5)
    print("evaluation completed")
    res.to_csv(OUTPUT_DCE_AMC, index=False)


def improve_example():
    data = transfer_data_from_dce_amc()
    print("data transformed")
    res = improve(llm, model_config, data, worker_count=5)
    print("improve completed")
    res.to_csv(OUTPUT_RAI, index=False)


def get_qags_data():
    data = []
    with open(os.path.join(FILE_DIR, SOURCE_FILE)) as f:
        raw_data = json.load(f)
        for item in raw_data[:5]:
            doc_id = item["doc_id"]
            article = item["source"]
            summary = item["system_output"]
            consistency_score = item["scores"]["consistency"]
            data.append(
                {
                    "id": doc_id,
                    "reference": article,
                    "candidate": summary,
                    "label": consistency_score,
                }
            )
        return data


def transfer_data_from_dce_amc():
    df = pd.read_csv(OUTPUT_DCE_AMC)
    cleaned_input = []

    def get_data(item):
        id, reference, score, dce_raw = (
            item["id"],
            item["reference"],
            item["score"],
            json.loads(item["dce_raw"]),
        )

        # only look at the ones that is inconsistent
        if score < 1:
            sentences = dce_raw["reason"]
            final_sentences = ""
            for sentence in sentences:
                reason = sentence["reason"]
                final_sentences += f"- sentence: {sentence} \n"
                final_sentences += f"  reason: {reason}\n"
            cleaned_input.append(
                {"id": id, "article": reference, "sentences": final_sentences}
            )

    df.apply(get_data, axis=1)
    return cleaned_input


if __name__ == "__main__":
    evalute_example()
    improve_example()
