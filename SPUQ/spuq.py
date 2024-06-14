from utils.perturbation import Paraphrasing, RandSysMsg, DummyToken, TemperaturePerturbation
from SPUQ.aggregation import IntraSampleAggregation, InterSampleAggregation
from utils.llm import LLM, LLM_OpenAI


class SPUQ:
    def __init__(self, llm: LLM, perturbation: str, aggregation: str, n_perturb: int):
        self.llm = llm
        assert (n_perturb > 0)

        if perturbation == 'paraphrasing':
            # Using gpt-3.5 for paraphrasing
            llm_perturb = LLM_OpenAI("gpt-35-turbo-v0301")
            self.perturbation = Paraphrasing(n_perturb, llm_perturb)
        elif perturbation == 'system_message':
            self.perturbation = RandSysMsg(n_perturb)
        elif perturbation == 'dummy_token':
            self.perturbation = DummyToken(n_perturb)
        elif perturbation == 'temperature':
            self.perturbation = TemperaturePerturbation(llm.T_min, llm.T_max, n_perturb)
        else:
            raise ValueError('Invalid perturbation method: %s' % perturbation)

        if aggregation in ['rouge1', 'rouge2', 'rougeL', 'sbert', 'bertscore']:
            self.aggregation = InterSampleAggregation(aggregation)
        elif aggregation in ['verbalized_word', 'verbalized_num']:
            self.aggregation = IntraSampleAggregation(llm, kind=aggregation)
        else:
            raise ValueError('Invalid aggregation method: %s' % aggregation)

    def run(self, messages: list, temperature: float):
        perturbed = self.perturbation.perturb(messages, temperature)
        inp_out = []
        outs = []
        for x, temperature in perturbed:
            model_config = {"temperature": temperature}
            out = self.llm.generate(x, **model_config)
            outs.append(out)
            inp_out.append((x, out))
        conf = self.aggregation.aggregate(inp_out)
        return {
            'perturbed': perturbed,
            'outputs': outs,
            'confidence': conf,
        }
