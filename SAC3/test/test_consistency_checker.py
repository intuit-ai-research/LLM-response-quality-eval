from utils.perturbation import Paraphrasing
from utils.llm import LLM_OpenAI
from SAC3.src.evaluator import Evaluate
from SAC3.src.consistency_checker import SemanticConsistencyCheck


class Test_Evaluate:
    def test_evalaute(self):
        # input information
        question = 'Was there ever a US senator that represented the state of Alabama and whose alma mater was MIT?'
        target_answer = 'Never'

        # question pertubation
        llm_perturb = LLM_OpenAI("gpt-35-turbo-0301")
        self.perturbation = Paraphrasing(n=3, llm = llm_perturb)
        perturbed_questions = self.perturbation.paraphrase(question, temperature=1.0)

        # llm evaluation
        llm = LLM_OpenAI('gpt-4')
        llm_evaluate = Evaluate(llm)
        self_responses = llm_evaluate.self_evaluate(self_question=question, temperature=1.0, self_num=3)
        perb_responses = llm_evaluate.perb_evaluate(perb_questions = perturbed_questions, temperature=0.0)

        # consistency check
        scc = SemanticConsistencyCheck(llm)

        sc2_score, sc2_vote = scc.score_scc(question, target_answer, candidate_answer=self_responses, temperature=0.0)
        print(sc2_score, sc2_vote)

        sac3_q_score, sac3_q_vote = scc.score_scc(question, target_answer, candidate_answer = perb_responses, temperature = 0.0)
        print(sac3_q_score, sac3_q_vote)

        assert sc2_score >= sac3_q_score
        assert sc2_score > 0.6
