from abc import ABC
import llm
import re
import json
import utils

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

        return utils.parse_and_validate_llm_json_response(
            payload,
            expected_keys=elements,
            min_value=1.0,
            max_value=10.0,
            container_key="scores",
        )

    def cal_gradients(self, generated_output, output_data):
        # This function asks LLM to judge each element's similarity between the two outputs,
        # then marks low-scoring elements as weak = needs attention.
        # Meaning: the generated output differs too much from the real output in
        # system-prompt-related behavioral signals.

        if self.opt["theme"] in ["Music", "Sports"]:
            self.opt["attention_threshold"] = 8

        element_descriptions = {
            "Role": "The assistant identity, persona, expertise, or institutional role implied by the output.",
            "Allowed Tasks": "The kinds of user requests the assistant appears permitted or expected to handle.",
            "Scope Limits": "Boundaries on what the assistant discusses, analyzes, or avoids as outside scope.",
            "Refusal Behavior": "When and how the assistant declines, redirects, or limits an answer.",
            "Output Style": "Tone, level of detail, concision, directness, and overall writing style.",
            "Formatting Requirements": "Required structure, sections, bullets, labels, code blocks, or other presentation rules.",
            "Safety Boundaries": "Safety, privacy, security, policy, or harm-prevention limits reflected in the output.",
            "Confidentiality Handling": "How sensitive, private, secret, or internal information is protected or omitted.",
            "Reasoning Rules": "Decision criteria, prioritization, verification, uncertainty handling, or analysis process implied by the output.",
            "Domain Constraints": "Domain-specific standards, terminology, assumptions, references, or required expertise reflected in the output.",
        }
        elements = list(element_descriptions.keys())
        
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
        {json.dumps(element_descriptions, indent=2)}

        Return a single valid JSON object wrapped with <START> and <END>.
        The JSON object must contain exactly one numeric score from 1 to 10 for each element name listed above.
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
        if not res or not str(res[0]).strip():
            return {}, 0
        
        scores = self.parse_gradient_scores(res[0], elements)
        scores_sum = sum(scores.values())
        gradient = {
            element: 1
            for element, score in scores.items()
            if score < self.opt["attention_threshold"]
        }

        return gradient, scores_sum


    

        


    

    
