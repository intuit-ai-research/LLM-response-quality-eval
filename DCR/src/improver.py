from typing import Dict, List

import pandas as pd

from utils.llm import LLM
from DCR.src.components.rai import RAI
from DCR.src.inputs.input import Input
from DCR.src.inputs.rai_summary_input import RAISummaryInput
from DCR.src.prompts.rai_summary_prompt import rai_summary_prompt


def improve(
    llm: LLM,
    model_config: Dict,
    input_data: List[Input],
    rai_prompt=rai_summary_prompt,
    worker_count=1,
) -> pd.DataFrame:
    orin_df = pd.DataFrame.from_dict(input_data)

    rai_inputs = []
    for item in input_data:
        rai_inputs.append(RAISummaryInput(item))

    rai = RAI(llm, model_config)
    rai_df = rai.improve(rai_prompt, rai_inputs, worker_count=worker_count)

    final_df = pd.merge(orin_df, rai_df, on="id", how="left")
    return final_df
