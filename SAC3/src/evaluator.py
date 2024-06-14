from utils.llm import LLM


class Evaluate:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.prompt_temp = "Answer the following question:\n"

    def self_evaluate(self, self_question, temperature, self_num):
        self_response = []
        prompt = self.prompt_temp + "\nQ:" + self_question
        model_config = {"temperature": temperature}
        # llm model: openai, open-source models (falcon, guanaco)
        for i in range(self_num):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            res = self.llm.generate(messages, **model_config)
            self_response.append(res)

        return self_response

    def perb_evaluate(self, perb_questions, temperature):
        perb_response = []
        for i in range(len(perb_questions)):
            prompt = self.prompt_temp + "\nQ:" + perb_questions[i]
            model_config = {"temperature": temperature}
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            res = self.llm.generate(messages, **model_config)
            perb_response.append(res)

        return perb_response
