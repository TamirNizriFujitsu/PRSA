from abc import ABC
import llm
import re
import json

class PromptOptimizer(ABC):
    def __init__(self, args, scorer, gradient_dict):
        self.opt = args
        self.scorer = scorer
        self.gradient_dict = gradient_dict

class PAA(PromptOptimizer):
    """ PAA: Prompt Attention Algorithm
        This idea of Prompt Attention Algorithm is borrowed from this paper: "Automatic Prompt Optimization with "Gradient Descent" and Beam Search"
    """
    

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts

    def filter_target_score(self, text, feedbacks):
        if feedbacks == []:
            numbers = re.findall(r'\d+\.\d+|\d+', text)
            target_score = [float(number) for number in numbers if float(number) < 10]
            assert len(target_score) == 1
        else:
            target_score = feedbacks
        return target_score[0]


    def extract_number(self, s):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        return numbers[0] if numbers else None

    def parse_gradient_scores(self, text, elements):
        tagged = self.parse_tagged_text(text, "<START>", "<END>")
        payload = tagged[0] if tagged else text

        try:
            scores = json.loads(payload)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", payload, re.DOTALL)
            if not json_match:
                raise
            scores = json.loads(json_match.group(0))

        if "scores" in scores and isinstance(scores["scores"], dict):
            scores = scores["scores"]

        return {
            element: float(self.extract_number(str(scores[element])))
            for element in elements
            if element in scores and self.extract_number(str(scores[element])) is not None
        }

    def cal_gradients(self, generated_output, output_data):
        # This function asks LLM to judge each element's similarity between the two outputs,
        # then marks low-scoring elements as weak = needs attention.
        # Meaning: the generated output differs too much from the real output in tone, audience, and structure.

        if self.opt["theme"] in ["Music", "Sports"]:
            self.opt["attention_threshold"] = 8

        elements = ['Characteristic','Topic','Argument','Structure','Style','Tone','Purpose','Sentence Type','Audience','Background']
        
        system_prompt = """
        You are an expert evaluator comparing two model outputs.
        Rate how similar the Generated Output is to the Real Output for each requested element.
        Use a score from 1 to 10 for each element. A score of 1 means very low similarity or no meaningful match for that element. Higher scores mean higher similarity, and 10 means identical or functionally equivalent for that element.
        Return only a JSON object wrapped with <START> and <END>.
        The JSON keys must be exactly the requested element names and the values must be numeric scores.
        """
        user_prompt = f"""
        Generated Output:
        "{generated_output}"

        Real Output:
        "{output_data}"

        Elements to score:
        {json.dumps(elements)}

        Return a single valid JSON object wrapped with <START> and <END>.
        The JSON object must contain exactly one numeric score from 1 to 10 for each element listed above.
        Use the element names exactly as the JSON keys.
        Do not include placeholders, comments, explanations, markdown, or any text outside the tags.
        """
        user_prompt = '\n'.join([line.lstrip() for line in user_prompt.split('\n')])
        res = llm.chatGPT_inference(
            system_prompt=system_prompt,
            text=user_prompt,
            model=self.opt.get("gradient_llm_model", "gpt-4o"),
            temperature=0.0,
            call_source="gradient",
        )
        scores = self.parse_gradient_scores(res[0], elements)
        scores_sum = sum(scores.values())
        gradient = {
            element: 1
            for element, score in scores.items()
            if score < self.opt["attention_threshold"]
        }

        return gradient, scores_sum


    

        


    

    
