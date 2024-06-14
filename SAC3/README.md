[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/github/license/intuit/email-decomposer)](https://raw.githubusercontent.com/intuit/email-decomposer/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# SAC3: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency [EMNLP2023]


## :fire: News
- [2024.02] SAC3 work will be presented in [AI for Production](https://home.mlops.community/home/events/ai-in-production-2024-02-15) organized by MLOps community!  
- [2023.12] SAC3 blog is published in [Intuit Engineering](https://medium.com/intuit-engineering/intuit-ai-research-debuts-novel-approach-to-reliable-hallucination-detection-in-black-box-language-746d7f720c50) 
- [2023.11] SAC3 code is released.
- [2023.11] SAC3 [arxiv link](https://arxiv.org/abs/2311.01740) is available.
- [2023.10] SAC3 paper is accepted and to appear at EMNLP 2023.


## 🤔 What is SAC3

Semantic-aware cross-check consistency (SAC3) is a novel sampling-based hallucination detection method that expands on the principle of self-consistency checking and incorporates additional mechanisms to detect both question-level and model-level hallucinations by leveraging advances including semantically equivalent question perturbation and cross-model response consistency checking. More details can be found in our paper [arxiv link](https://arxiv.org/abs/2311.01740).

![](assets/overview_sac3.png)


**Key observation**: Solely checking the self-consistency of LLMs is not sufficient for deciding factuality. Left: generated responses to the same question may be consistent but non-factual. Right: generated responses may be inconsistent with the original answer that is factually correct.

![](assets/key_observation.png)


## 🤖 Installation


#### Requirements 

- python 3.8
- openai > 1.25.0

- Create env and download all the packages required as follows: 

```
conda create -n sac3 python=3.8
source activate sac3
pip install -r requirements.txt
```


SAC3  can be installed from pip (coming soon!):

```bash
pip install sac3
```


## 🚀 Quickstart
The easiest way to start playing is
1. [```Jupyter Notebook```](notebook/quick_start.ipynb)
2. Directly run ```python main.py```
3. Google Colab (coming soon)

Note - Use your llm apikey in `utils\llm.py` to make llm calls.

## 📃 Usage

### Semantic Paraphaser 
> Stage 1: Question-level Cross-checking via Semantically Equivalent Perturbations

``` python
from utils.perturbation import Paraphrasing
from utils.llm import LLM_OpenAI

# input information - user question and target answer 
question = 'Is 3691 a prime number?'
target_answer = 'No, it is not a prime number'

# question pertubation
llm_perturb = LLM_OpenAI("gpt-35-turbo-0301")
perturbation = Paraphrasing(n=5, llm = llm_perturb)
perturbed_questions = perturbation.paraphrase(question, temperature=1.0)


```
The output is 
``` 
perturbed_questions = ['Can 3691 only be divided by 1 and itself?',
 'Are there any factors of 3691 other than 1 and itself?',
 'Does 3691 belong to the set of prime numbers?',
 'Is 3691 indivisible except by 1 and itself?',
 'Is 3691 not evenly divisible by any number other than 1 and itself?']

```
### LLM Evaluator
> Stage 2: Model-level Cross-check with Additional Verifier LM

```python
from SAC3.src.evaluator import Evaluate
from utils.llm import LLM_OpenAI

# call target model, e.g., gpt-3.5 or gpt-4
llm = LLM_OpenAI('gpt-4')
llm_evaluate = Evaluate(llm)

# sampled multiple responses given the original question 
self_responses = llm_evaluate.self_evaluate(self_question = question, temperature = 1.0, self_num = 5)

# sampled multiple responses given perturbed question
perb_responses = llm_evaluate.perb_evaluate(perb_questions = perturbed_questions, temperature=0.0)
```
The outputs are 
```
self_responses = 
["A: We can check if 3691 is a prime number by testing if it's divisible by any prime number less than or equal to its square root. \n\nThe square root of 3691 is approximately 60.77, so we only need to check if it's divisible by prime numbers less than or equal to 60. \n\nWe can quickly eliminate even numbers, since 3691 is odd. \n\nDivisibility by 3:\n3 + 6 + 9 + 1 = 19, which is not divisible by 3. Therefore, 3691 is not divisible by 3. \n\nDivisibility by 5:\n3691 does not end in 5 or 0, so it's not divisible by 5. \n\nDivisibility by 7:\nTo check if 3691 is divisible by 7, we can take the last digit (1) and double it, then subtract that result from the remaining digits (369 - 2 = 367). We can continue this process until we get a number we know the divisibility of. \n\n36 - 2*1 = 34\n3 - 2*4 = -5\n\nSince -5 is not divisible by 7, neither is 3691. \n\nWe can continue checking for divisibility by other prime numbers, but it's clear that 3691 is not divisible by any primes less than or equal to 60. Therefore, 3691 is a prime number.",
 'Yes, 3691 is a prime number.',
 'A: Yes, 3691 is a prime number.',
 'No, 3691 is not a prime number. It is divisible by 7 and 17.',
 'A: Yes, 3691 is a prime number.']
```
```
perb_responses = 
['As an AI language model, I can say that 3691 can only be divided by 1 and itself, as it is a prime number.',
 'As an AI language model, I do not have the ability to manually factorize numbers. However, I can tell you that if there are any factors of 3691 other than 1 and itself, then 3691 would not be a prime number.',
 'A: No, 3691 is not a prime number as it is divisible by 7 and 527.',
 'A: Yes, 3691 is indivisible except by 1 and itself, making it a prime number.',
 'As an AI language model, I can calculate that 3691 is not evenly divisible by any number other than 1 and itself. Therefore, it is a prime number.']
 ```
### Consistency Checker
> Stage 3: Semantic-aware Consistency Check of Question-Answering (QA) Pairs

-  Self-checking Consistency Score
```python
from SAC3.src.consistency_checker import SemanticConsistencyCheck

# call LLM to evaluate if the sampled responses are semantically equivalent to the target answer
scc = SemanticConsistencyCheck(llm)

# self-consistency check 
sc2_score, sc2_vote = scc.score_scc(question, target_answer, candidate_answer = self_responses, temperature = 0.0)
print(sc2_score, sc2_vote)
```
The output is 
```
0.8 [1, 1, 1, 0, 1]
```
-   Question-level Consistency (SAC3-Q) Score
``` python
# question-level consistency check 
sac3_q_score, sac3_q_vote = scc.score_scc(question, target_answer, candidate_answer = perb_responses, temperature = 0.0)
print(sac3_q_score, sac3_q_vote)
```
The output is 
```
0.6 [1, 0, 0, 1, 1]
```

### Illustrative Example 

An illustrative example of self-consistency, cross-question consistency, and cross-model consistency check.
The original question and answer are “Is 3691 a prime number?” and “No, 3691 is not a prime number.
It is divisible by 7 and 13”, respectively. Each row presents a set of sampled QA pairs along with its
consistency regarding the original answer, and the predicted factuality of the original answer. 

![](assets/example.png)


## 💁Citation 

```
@inproceedings{zhang-etal-2023-sac3,
    title = "SAC3: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency",
    author = "Zhang, Jiaxin  and
      Li, Zhuohang  and
      Das, Kamalika  and
      Malin, Bradley  and
      Kumar, Sricharan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.1032",
    doi = "10.18653/v1/2023.findings-emnlp.1032",
    pages = "15445--15458"
}
```


