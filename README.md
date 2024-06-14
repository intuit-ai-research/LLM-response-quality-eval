# LLM-response-quality-eval

LLM-response-quality-eval is a research initiative by Intuit AI Research that focuses on fundamental research in the area of LLM response quality assessment. Some of our projects include -
- Hallucination detection and mitigation :- [(SAC3)](#sac3-reliable-hallucination-detection-in-black-box-language-models-via-semantic-aware-cross-check-consistency)Semantic-aware cross-check consistency, [DCR-Consistency](#dcr-consistency--divide-conquer-reasoning-for-consistency-evaluation-and-improvement-of-large-language-models): Divide-Conquer-Reasoning for Consistency Evaluation and Improvement of Large Language Models
- Uncertainty quantification :- [SPUQ](#spuq--perturbation-based-uncertainty-quantification-for-large-language-models): Perturbation-Based Uncertainty Quantification for Large Language Models

## :fire: News
- [2024.04] [Intuit presents innovative approach to Quantify LLM Uncertainty at EACL 2024](https://medium.com/intuit-engineering/intuit-presents-innovative-approach-to-quantifying-llm-uncertainty-at-eacl-2024-f839a8f1b89b)
- [2024.04] SPUQ [arxiv link](https://arxiv.org/abs/2403.02509) available
- [2024.02] SAC3 work presented in [AI for Production](https://home.mlops.community/home/events/ai-in-production-2024-02-15) organized by MLOps community! 
- [2024.01] DCR-consistency [arxiv link](https://arxiv.org/abs/2401.02132) available.
- [2023.12] [Intuit AI Research Debuts Novel Approach to Reliable Hallucination Detection in Black Box Language Models at EMNLP 2023](https://medium.com/intuit-engineering/intuit-ai-research-debuts-novel-approach-to-reliable-hallucination-detection-in-black-box-language-746d7f720c50) 
- [2023.11] SAC3 [arxiv link](https://arxiv.org/abs/2311.01740) available.
- [2023.10] SAC3 paper accepted at EMNLP 2023.

## Hallucination detection and mitigation
### SAC3: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency
[[Paper]](https://arxiv.org/abs/2311.01740) [[Code]](https://github.com/intuit-ai-research/LLM-response-quality-eval/tree/master/SAC3)

Semantic-aware cross-check consistency (SAC3) is a novel sampling-based hallucination detection method that expands on the principle of self-consistency checking and incorporates additional mechanisms to detect both question-level and model-level hallucinations by leveraging advances including semantically equivalent question perturbation and cross-model response consistency checking.

### DCR-consistency : Divide-Conquer-Reasoning for Consistency Evaluation and Improvement of Large Language Models
[[Paper]](https://arxiv.org/abs/2401.02132) [[Code]](https://github.com/intuit-ai-research/LLM-response-quality-eval/tree/master/DCR)

DCR-Consistency is a novel framework that uses LLM agents to detect and mitigate inconsistencies, or in other words hallucinations. It takes advantage of LLM's power in semantic understanding while circumventing known pitfalls such as relatively poor performance in math. For more details please see [our paper](https://arxiv.org/pdf/2401.02132.pdf).
Given a `reference` as the ground truth and a `candidate` to evaluate, it will output a numeric score between [0, 1] indicating its consistency where 0 means no sentence in the `candidate` is consistent and 1 otherwise. It also outputs a list of `reasons` about why this score is generated. Better yet, based on such `reasons`, it can improve the `candidate` and mitigate detected inconsistencies. 

## Uncertainty quantification
### SPUQ : Perturbation-Based Uncertainty Quantification for Large Language Models
[[Paper]](https://arxiv.org/abs/2403.02509) [[Code]](https://github.com/intuit-ai-research/LLM-response-quality-eval/tree/master/SPUQ)

SPUQ is an LLM uncertainty calibration algorithm. It provides a confidence score for each query, for a given LLM. Experiments show that this confidence score is correlated with the generation accuracy, and therefore provides a useful LLM response evaluation metric on-the-fly.

The details of the approach are documented in our [paper](https://arxiv.org/abs/2403.02509) published at EACL-2024 Conference.

The basic idea is to check whether an LLM provides a significantly different answer when we ask the same question in a slightly different way. If it does, we assume the LLM is not confident in this case. SPUQ perturbs the input (including the prompt and the temperature) to get multiple outputs, and then aggregate the outputs to obtain the final confidence score. This allows SPUQ to address both epistemic (via perturbation) and aleatoric (via sampling) uncertainties, and it provides better calibration than some of the other existing methods.

