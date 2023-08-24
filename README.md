---
title: LLM Challenge
emoji: 💬
colorFrom: red
colorTo: yellow
sdk: docker
pinned: true
---

# Technical challenge
Author: Pau Rodriguez Inserte (@pauri32)

## Setup and running instructions
You have two options to run the challenge!
### HuggingFace Space
1. Just click on the 'App' tab of the HuggingFace space.
2. Use FastAPI's interactive Swagger UI
3. Go to ```/language-detection``` endpoint and introduce an input text to identify its language (only available English, French and Spanish)
4. Go to ```/entity-recognition``` endpoint and introduce an input text to retrieve and count locations, people and organizations named-entities.
### Download Docker image
1. You can run it locally by using the following command ```docker run -it -p 7860:7860 --platform=linux/amd64 registry.hf.space/pauri32-llm-challenge:latest```
2. By opening ```localhost:7860``` in your browser, you will be able to interact with FastAPI's UI.

## Reasoning of the LLM design choices
### Model selected: BloomZ
The model selected for this project is BloomZ with 1.1B parameters. The size of the model was decided according to my hardware limitations (this is the largest model I could fit without a GPU, 4bit and 8bit quantizing is not possible on CPU). The type of model used has been chosen considering the following characteristics: 
* BloomZ is a model fine-tuned on instructions with Bloom as base model. Bloom was trained on 46 different languages, this is highly relevant for the language detection task. Finding a model trained on a smaller subset of languages may have been a good option, but with BloomZ it will be easier to scale to a highly multilingual classification.
* The main language in which the model has been trained is English. Therefore it can still be strong for the entity recognition task. This is also the reason why the instructions are in English.
* The model has a wide range of sizes, up to 176B. In case of having a good GPU, it would be easy to increase the size of the model without many further modifications.
* A model fine-tuned to follow instructions will have a better performance for unseen tasks. In this case, since the model has not been fine-tuned for the specific tasks targeted, good zero and few-shot performance are important.

### Task 1 design: Language detection
* The model receives the instruction of identifying if the language of the string is English, Spanish or French.
* The template follows the format <input_sentence>(language_id), selected after a quick prompt engineering process.
* The languages are identified with 'english' for English, 'español' for Spanish and 'française' for French. My reasoning behind this decision is the hypothesis that the model, because it has been trained on these languages, will be more likely to keep the language of the previous tokens for the next generation, since in training sequences rarely switch languages within a sentence. So just the fact of having the language names in that language, helps with the classification.
* 3 shots are added with one sentence for each language (shots should be more curated for a production application).
* If none of the language identifiers is detected, the language is classificated as unknown. This is not the ideal scenario, but a method to avoid it is later suggested.
### Task 2 design: Name entity recognition
* The model is asked to identify entities related to locations, people and organizations.
* The LLM generates the entities in the following format <entity>(entity_type), selected after a quick prompt engineering process.
* Asking the model to identify the type of entity, even when it is not needed, is a way to decompose the problem, something similar to a chain-of-thought (COT). The model might not be familiar with the concept of named entity, but it is with locations, people and organizations. In the end, this step was helpful for the model after evaluating some prompts.
* Finally, the counting of the entities is done by scanning the generated string with regex. There is no need to ask the model to count them and add complexity!
* 3 shots were added to the prompt to improve performance, showing the model the style and all the entity types considered.

## Further improvements
The most important improvement would be to fine-tune the model on the specific tasks. To do this, I would follow the following steps:
* For the language detection task, any collection of documents of the targeted languages would be useful (if we know in which language is every document). These documents would be split at sentence-level and I would create a dataset of instructions with the same format as the current shots.
* For the second task, the best option would be to find an existing dataset for this task, such as [CONLLPP](https://huggingface.co/datasets/conllpp). With this dataset and the prompt template designed, we could generate a dataset. 
* Another option for the second task, in case there was no dataset for the task (for example if we want to target a specific domain or do it in a different language with less resources), we could generate the dataset by 'distilling' information from another LLM. For example, if this task required entity recognition in Catalan and there was no dataset, we could infer a few examples from a bigger model like GPT4. However, this solution has three main drawbacks: (1) we may have to pay for the models; (2) their license not always allows this; (3) even the best models make mistakes, we would be assuming some error in our dataset.

Fine-tuning the models for specific tasks would make the model perform better on them. Another variation we could do is to remove the few-shots used in the current code. Despite curated shots are usually helpful for the model, they also increase inference time. It should be studied if the improvement with shots is relevant compared to a higher latency (this would depend on the specifications of the project).
A third improvement would be to use 'forced decoding' for the classification task. By this, we could make sure one of the 3 language identifiers are generated and the answer would never be 'unknown'. During fine-tuning, the model could be instructed to generate directly the codes 'en', 'es', 'fr'.
Finally, for the next project I would like to try Gradio, since I have seen that it's a nice way to build interfaces to test the models.

## Evaluation proposal
For the language detection and NER tasks, the best metric for evaluation would be to compute the weighted F1-score (also possible to check the precision, recall and confusion matrix for better insights). 
In order to evaluate LLMs, I would use EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), which I'm currently using for other projects and highly recommend. It's also the one running behind the well-known Open LLM Leaderboard. With this framework it would be very easy to create a custom task and evaluate with the metrics described above. For example, it would be very easy to do a forced decoding evaluation as I suggested for the language detection task, the framework gives you the logits associated to the candidate tokens and you just pick the maximum in every case.
