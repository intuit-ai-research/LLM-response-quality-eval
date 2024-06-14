from utils.llm import LLM

class SemanticConsistencyCheck:
    def __init__(self, llm:LLM):
        self.llm = llm
        self.prompt_temp = """
        Are the following two Question-Answer(QA) pairs semantically equivalent? 
        Provide your best guess and the probability that it is correct (0.0 to 1.0).
        Given ONLY the guess (Yes or No) and probability, no other words or explanation. 
        For example:
        Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!> 
        Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; 
        just the probability!>
        """

    def score_scc(self, question, target_answer, candidate_answer, temperature):
        sc_output = []
        target_pair = "Q:" + question + "\nA:" + target_answer
        num_candidate_answer = len(candidate_answer)
        for i in range(num_candidate_answer):
            candidate_pair = "Q:" + question + "\nA:" + candidate_answer[i]
            prompt = (
                self.prompt_temp
                + "\nThe first QA pair is:\n"
                + target_pair
                + "\nThe second QA pair is:\n"
                + candidate_pair
            )
            messages = [
                           {"role": "system", "content": "You are a helpful assistant."},
                           {"role": "user", "content": prompt},
                       ]

            model_config = {"temperature" : temperature}
            res = self.llm.generate(messages, **model_config)

            guess = res.split(":")[1].split("\n")[0].strip()
            print(res, guess)
            value = 0 if guess == "Yes" else 1
            print("value", value)
            sc_output.append(value)

        score = sum(sc_output) / num_candidate_answer
        return score, sc_output
