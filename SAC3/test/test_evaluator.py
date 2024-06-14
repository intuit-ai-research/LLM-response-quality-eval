from utils.perturbation import Paraphrasing
from utils.llm import LLM_OpenAI
from SAC3.src.evaluator import Evaluate

class Test_Evaluate:
    def test_evalaute(self):
        # input information
        question = 'Was there ever a US senator that represented the state of Alabama and whose alma mater was MIT?'

        # question pertubation
        llm_perturb = LLM_OpenAI("gpt-35-turbo-0301")
        self.perturbation = Paraphrasing(n=3, llm = llm_perturb)
        perturbed_questions = self.perturbation.paraphrase(question, temperature=1.0)

        # llm evaluation
        llm = LLM_OpenAI('gpt-4')
        llm_evaluate = Evaluate(llm)
        self_responses = llm_evaluate.self_evaluate(self_question=question, temperature=1.0, self_num=3)
        perb_responses = llm_evaluate.perb_evaluate(perb_questions = perturbed_questions, temperature=0.0)

        assert len(self_responses) == 3
        assert len(perb_responses) == 3

