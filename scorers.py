import time
import numpy as np
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import jensenshannon
from collections import Counter
import re
import os
from transformers import BertTokenizer, BertModel
import torch
import fkassim.FastKassim as fkassim
import sentence_bert


java_path = "./tool/jdk-21.0.1"
#os.environ['JAVAHOME'] = java_path
os.environ['JAVA_HOME'] = java_path

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


class MetricsScorer:
    def __init__(self, evaluator, m):
        self.evaluator = evaluator
        self.m = m
        # Reuse parser instance and parsed trees across repeated pairwise scoring calls.
        self._fastkassim = fkassim.FastKassim(fkassim.FastKassim.LTK)
        self._syntactic_parse_cache = {}

    def eval_pred_target(self, eval_type, pred_outputs, target_outputs):
        sum_similarity_pred = 0
        for pred_output in pred_outputs:
            for target_output in target_outputs:
                sum_similarity_pred += self._compute_scores(eval_type, pred_output, target_output)
        average_similarity_pred = sum_similarity_pred / (len(pred_outputs) * len(target_outputs))
        return average_similarity_pred

    def eval_target_target(self, eval_type, target_outputs):
        if self.m <= 1:
            raise ValueError(f"`m` must be at least 2 to compute pairwise similarity. Got m={self.m}")
    
        sum_similarity_ori = 0
        for i in range(len(target_outputs)):
            for j in range(i + 1, len(target_outputs)):
                sum_similarity_ori += self._compute_scores(eval_type, target_outputs[i], target_outputs[j])
        
        num_pairs = len(target_outputs) * (len(target_outputs) - 1) / 2
        average_similarity_ori = sum_similarity_ori / num_pairs
        return average_similarity_ori


    def evaluate_target_prompt(self, fn, inputs, target_prompt):
 
        target_semantic_score_list = []
        # target_syntactic_score_list = []
        target_js_score_list = []
        target_outputs_per_input = {}
        for i, input_data in enumerate(inputs):
            # timing check 
            start = time.time()
            target_outputs = [fn(input_data, target_prompt) for _ in range(self.m)]
            print(f"**** Time for generating target outputs***** {time.time() - start:.3f}")
            # for reusing in the evaluate_stolen_prompt function
            target_outputs_per_input[f"input{i}"] = target_outputs
            if any(out is None or out.strip() == "" for out in target_outputs):
                return None, None

            # timing check
            start = time.time()
            target_semantic_score = self.eval_target_target("semantic_similarity", target_outputs)
            print(f"**** Total time for semantic similarity calculation - only target outputs phase ***** {time.time() - start:.3f}")
            # Disabled for speed test:
            # target_syntactic_score = self.eval_target_target("syntactic_similarity", target_outputs)
            # timing check
            start = time.time()
            target_js_score = self.eval_target_target("jensen_shannon_divergence", target_outputs)
            print(f"**** Total time for structure similarity calculation - only target outputs phase ***** {time.time() - start:.3f}")

            target_semantic_score_list.append(target_semantic_score)
            # target_syntactic_score_list.append(target_syntactic_score)
            target_js_score_list.append(target_js_score)
 
        avg_target_semantic_score = np.mean(target_semantic_score_list)
        # Disabled for speed test:
        avg_target_syntactic_score = None
        avg_target_js_score = np.mean(target_js_score_list)

        avg_target_score = [avg_target_semantic_score,  avg_target_syntactic_score, avg_target_js_score]

        return avg_target_score, target_outputs_per_input
    


    def evaluate_stolen_prompt(self, fn, inputs, target_prompt, stolen_prompt, target_outputs_per_input):
 
        semantic_score_list = []
        # syntactic_score_list = []
        js_score_list = []

        
        for i, input_data in enumerate(inputs):
            # timing check
            start = time.time()
            pred_outputs = [fn(input_data, stolen_prompt) for _ in range(self.m)]
            print(f"**** Time for generating stolen outputs***** {time.time() - start:.3f}")
            if any(out is None or out.strip() == "" for out in pred_outputs):
                return None

            # timing check
            start = time.time()
            pred_semantic_score = self.eval_pred_target("semantic_similarity", pred_outputs, target_outputs_per_input[f"input{i}"])
            print(f"**** Total time for semantic similarity calculation - target vs stolen output phase ***** {time.time() - start:.3f}")
            # Disabled for speed test:
            # pred_syntactic_score = self.eval_pred_target("syntactic_similarity", pred_outputs, target_outputs)
            # timing check
            start = time.time()
            pred_js_score = self.eval_pred_target("jensen_shannon_divergence", pred_outputs, target_outputs_per_input[f"input{i}"])
            print(f"**** Total time for structure similarity calculation - target vs stolen output phase ***** {time.time() - start:.3f}")

            semantic_score_list.append(pred_semantic_score)
            # syntactic_score_list.append(pred_syntactic_score)
            js_score_list.append(pred_js_score)
           
        avg_semantic_score = np.mean(semantic_score_list)
        # Disabled for speed test:
        avg_syntactic_score = None
        avg_js_score = np.mean(js_score_list)

        avg_score = [avg_semantic_score, avg_syntactic_score, avg_js_score]

        return avg_score


    def sliding_window_tokenize(self, text, tokenizer, max_length, stride):
        """
        Sliding window segmentation and Tokenization of text
        """
        tokens = tokenizer.tokenize(text)
        windows = []

        for i in range(0, len(tokens), stride):
            window = tokens[i:i + max_length]
            windows.append(window)

        return [tokenizer.convert_tokens_to_ids(window) for window in windows]

    def process_windows(self, windows, model):
        """
        Process each window and return the merged result
        """
        all_embeddings = []

        for window in windows:
            inputs = torch.tensor(window).unsqueeze(0)
            outputs = model(inputs)
            sentence_embedding = outputs.pooler_output
            all_embeddings.append(sentence_embedding)

        return torch.mean(torch.stack(all_embeddings), dim=0)
    
    

    def semantic_score_sbert(self, text1, text2):
        score = sentence_bert.calculate_similarity_sbert(text1, text2)
        return score

    def syntactic_score(self, doc1, doc2):
        if doc1 in self._syntactic_parse_cache:
            doc1_parsed = self._syntactic_parse_cache[doc1]
        else:
            doc1_parsed = self._fastkassim.parse_document(doc1)
            self._syntactic_parse_cache[doc1] = doc1_parsed

        if doc2 in self._syntactic_parse_cache:
            doc2_parsed = self._syntactic_parse_cache[doc2]
        else:
            doc2_parsed = self._fastkassim.parse_document(doc2)
            self._syntactic_parse_cache[doc2] = doc2_parsed

        score = self._fastkassim.compute_similarity_preparsed(doc1_parsed, doc2_parsed)
        return score


    def js_preprocess_text(self, text):
        return re.sub('[^a-z]+', ' ', text.lower()).split()

    def get_probability_distribution(self, text, vocab):
        frequency = Counter(text)
        return [frequency[word] / len(text) for word in vocab]
    
    def jensen_shannon_divergence(self, text1, text2, base=2):
        vocab = set(text1.split()) | set(text2.split())
        freq_dist1 = {word: text1.split().count(word) for word in vocab}
        freq_dist2 = {word: text2.split().count(word) for word in vocab}
        sum_freq1 = sum(freq_dist1.values())
        sum_freq2 = sum(freq_dist2.values())
        prob_dist1 = [freq_dist1[word] / sum_freq1 for word in vocab]
        prob_dist2 = [freq_dist2[word] / sum_freq2 for word in vocab]
        return jensenshannon(prob_dist1, prob_dist2, base=base)
    
    def norm_js_score(self, pred_score, target_score, max_diff=1):
        diff = abs(pred_score - target_score)
        if diff > max_diff:
            return 0
        return 1 - (diff / max_diff)
    
    def _compute_scores(self, eval_type, pred_output, target_output):
        # timing check
        start = time.time()
        if eval_type == "semantic_similarity":
            similarity_score = self.semantic_score_sbert(pred_output, target_output)
        elif eval_type == "syntactic_similarity":
            similarity_score = self.syntactic_score(pred_output, target_output)
        elif eval_type == "jensen_shannon_divergence":
            similarity_score = 1 / self.jensen_shannon_divergence(pred_output, target_output)
        else:
            raise ValueError(f"Unsupported evaluator type: {self.evaluator}")
        print(f"****{eval_type} Score Calculation***** - {time.time() - start:.3f}")
        return similarity_score
    
    #TODO
    def calculate_aggregated_scores(all_ranking_arrays):
        pass

    #TODO    
    def calculate_diff_scores(all_ranking_arrays):
        pass

    #TODO    
    def calculate_asr(category, target_model, scenario_suffix):
        # This functin should return the edited stolen prompt of the chosen category and its ASR
        #pass
        return "Hi", 0.9
