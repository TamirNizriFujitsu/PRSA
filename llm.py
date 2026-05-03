from abc import ABC, abstractmethod
import time
import config
import string
import json
import utils
import openai
import sys
import os
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


_MODEL_CATALOG_CACHE = None


class Predictor(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def inference(self, input, prompt):
        pass


class ChatGPTPredictor(Predictor):
    def inference(self, input, prompt, gpt_model=None):
        if gpt_model is None:
            gpt_model = self.config.get("target_llm_model", "CommandA")

        responses = chatGPT_inference(
            prompt,
            input,
            n=1,
            model=gpt_model,
            temperature=self.config["temperature"],
        )

        if not responses:
            print("[Warning] chatGPT_inference returned an empty response.")
            return None
        return responses[0]


class BatchSizeException(Exception):
    pass


def _load_model_catalog():
    global _MODEL_CATALOG_CACHE
    if _MODEL_CATALOG_CACHE is not None:
        return _MODEL_CATALOG_CACHE

    catalog_path = os.getenv("LLM_CATALOG_PATH", "catalog.json")
    if not os.path.exists(catalog_path):
        _MODEL_CATALOG_CACHE = {}
        return _MODEL_CATALOG_CACHE

    with open(catalog_path, "r") as f:
        _MODEL_CATALOG_CACHE = json.load(f)
    if not isinstance(_MODEL_CATALOG_CACHE, dict):
        raise ValueError(f"Model catalog at {catalog_path} must be a JSON object.")
    return _MODEL_CATALOG_CACHE


def _parse_azure_api_base(api_base):
    parsed = urlparse(api_base)
    if not parsed.scheme or not parsed.netloc:
        return api_base, None, None

    deployment = None
    marker = "/openai/deployments/"
    if marker in parsed.path:
        suffix = parsed.path.split(marker, 1)[1]
        deployment = suffix.split("/", 1)[0]

    query = parse_qs(parsed.query)
    api_version = query.get("api-version", [None])[0]
    normalized_base = f"{parsed.scheme}://{parsed.netloc}"
    return normalized_base, deployment, api_version


def _resolve_azure_deployment_from_env(model_name):
    model_key = model_name.upper().replace("-", "_").replace(".", "_")
    return (
        os.getenv(f"AZURE_OPENAI_DEPLOYMENT_{model_key}")
        or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or model_name
    )


def _build_azure_ai_chat_url(api_base, api_version=None):
    parsed = urlparse(api_base)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"Invalid azure_ai api_base: {api_base}")

    path = parsed.path or ""
    if "chat/completions" not in path:
        path = path.rstrip("/") + "/v1/chat/completions"

    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if api_version and "api-version" not in query_items:
        query_items["api-version"] = api_version

    query = urlencode(query_items)
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def _build_azure_ai_endpoint_url(api_base, endpoint_path=None, api_version=None):
    parsed = urlparse(api_base)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"Invalid azure_ai api_base: {api_base}")

    path = parsed.path or ""
    if not any(marker in path for marker in ("/chat/completions", "/v1/messages")):
        resolved_endpoint_path = endpoint_path or "/v1/chat/completions"
        path = path.rstrip("/") + resolved_endpoint_path

    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if api_version and "api-version" not in query_items:
        query_items["api-version"] = api_version

    query = urlencode(query_items)
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def _resolve_model_runtime(model_name):
    if not isinstance(model_name, str) or not model_name.strip():
        raise RuntimeError("Model name must be a non-empty string.")
    model_name = model_name.strip()

    catalog = _load_model_catalog()
    entry = catalog.get(model_name)
    if entry is None:
        allow_fallback = os.getenv("ALLOW_LEGACY_MODEL_FALLBACK", "0") == "1"
        if not allow_fallback:
            available = ", ".join(sorted(catalog.keys())) if catalog else "(empty catalog)"
            raise RuntimeError(
                f"Model '{model_name}' not found in catalog.json. Add it to catalog.json. Available models: {available}"
            )
        entry = {}

    provider = entry.get("provider")

    if provider == "azure":
        api_key = entry.get("api_key") or os.getenv(entry.get("api_key_env", "AZURE_OPENAI_API_KEY"))
        api_base = entry.get("api_base") or os.getenv("AZURE_OPENAI_API_BASE") or os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = entry.get("deployment")
        api_version = entry.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION")

        if api_base:
            api_base, deployment_from_url, version_from_url = _parse_azure_api_base(api_base)
            deployment = deployment or deployment_from_url
            api_version = api_version or version_from_url

        deployment = deployment or model_name
        api_version = api_version or "2024-02-15-preview"

        if not api_key:
            raise RuntimeError(
                f"No API key for Azure model '{model_name}'. Set {entry.get('api_key_env', 'AZURE_OPENAI_API_KEY')} or provide api_key in catalog.json."
            )
        if not api_base:
            raise RuntimeError(
                f"No api_base for Azure model '{model_name}'. Set it in catalog.json or AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_BASE."
            )

        return {
            "provider": "azure",
            "api_key": api_key,
            "api_base": api_base,
            "api_version": api_version,
            "deployment": deployment,
        }

    if provider == "openai":
        api_key = entry.get("api_key") or os.getenv(entry.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            raise RuntimeError(
                f"No API key for OpenAI model '{model_name}'. Set {entry.get('api_key_env', 'OPENAI_API_KEY')} or provide api_key in catalog.json."
            )
        return {
            "provider": "openai",
            "api_key": api_key,
            "model": entry.get("model", model_name),
        }

    if provider == "azure_ai":
        api_key = entry.get("api_key") or os.getenv(entry.get("api_key_env", "AZURE_AI_API_KEY"))
        api_base = entry.get("api_base")
        api_version = entry.get("api_version") or os.getenv("AZURE_AI_API_VERSION")
        model_alias = entry.get("model", model_name)
        endpoint_path = entry.get("endpoint_path")
        request_format = entry.get("request_format", "openai_chat")
        request_headers = entry.get("request_headers", {})

        if not api_key:
            raise RuntimeError(
                f"No API key for azure_ai model '{model_name}'. Set {entry.get('api_key_env', 'AZURE_AI_API_KEY')} or provide api_key in catalog.json."
            )
        if not api_base:
            raise RuntimeError(
                f"No api_base for azure_ai model '{model_name}'. Set it in catalog.json."
            )

        endpoint = entry.get("endpoint") or _build_azure_ai_endpoint_url(
            api_base,
            endpoint_path=endpoint_path,
            api_version=api_version,
        )

        return {
            "provider": "azure_ai",
            "api_key": api_key,
            "endpoint": endpoint,
            "model": model_alias,
            "auth_header": entry.get("auth_header", "api-key"),
            "auth_prefix": entry.get("auth_prefix", ""),
            "request_format": request_format,
            "request_headers": request_headers,
        }

    # Optional legacy fallback for ad-hoc model usage without catalog entries.
    # Disabled by default; enable only with ALLOW_LEGACY_MODEL_FALLBACK=1.
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_base = os.getenv("AZURE_OPENAI_API_BASE") or os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_base:
        azure_base, deployment_from_url, version_from_url = _parse_azure_api_base(azure_base)
        return {
            "provider": "azure",
            "api_key": azure_key,
            "api_base": azure_base,
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION") or version_from_url or "2024-02-15-preview",
            "deployment": _resolve_azure_deployment_from_env(model_name) or deployment_from_url,
        }

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return {
            "provider": "openai",
            "api_key": openai_key,
            "model": model_name,
        }

    raise RuntimeError(
        f"Model '{model_name}' not found in catalog.json and no fallback API credentials were found in environment."
    )


def _create_chat_completion(messages, model, temperature, n, top_p, max_tokens):
    runtime = _resolve_model_runtime(model)
    is_gpt5_family = isinstance(model, str) and model.startswith("gpt-5")
    token_param = (
        {"max_completion_tokens": max_tokens}
        if is_gpt5_family
        else {"max_tokens": max_tokens}
    )
    sampling_params = {}
    if is_gpt5_family:
        # GPT-5 endpoints may reject non-default temperature values.
        if top_p is not None:
            sampling_params["top_p"] = top_p
    else:
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }

    if runtime["provider"] == "azure":
        openai.api_type = "azure"
        openai.api_key = runtime["api_key"]
        openai.api_base = runtime["api_base"]
        openai.api_version = runtime["api_version"]

        return openai.ChatCompletion.create(
            engine=runtime["deployment"],
            n=n,
            messages=messages,
            timeout=(300, 300),
            **sampling_params,
            **token_param,
        )

    if runtime["provider"] == "openai":
        openai.api_key = runtime["api_key"]
        return openai.ChatCompletion.create(
            model=runtime["model"],
            n=n,
            messages=messages,
            timeout=(300, 300),
            **sampling_params,
            **token_param,
        )

    if runtime["provider"] == "azure_ai":
        request_format = runtime.get("request_format", "openai_chat")
        if request_format == "anthropic_messages":
            system_chunks = []
            anthropic_messages = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    if isinstance(content, str) and content:
                        system_chunks.append(content)
                    continue
                if role in ("user", "assistant"):
                    anthropic_messages.append({"role": role, "content": content})

            payload = {
                "model": runtime["model"],
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
            }
            if system_chunks:
                payload["system"] = "\n\n".join(system_chunks)
            # Some Anthropic-backed endpoints reject requests that specify
            # both temperature and top_p at the same time.
            if temperature is not None:
                payload["temperature"] = temperature
            elif top_p is not None:
                payload["top_p"] = top_p
        else:
            payload = {
                "model": runtime["model"],
                "messages": messages,
                "temperature": temperature,
                "n": n,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        auth_header = runtime.get("auth_header", "api-key")
        auth_prefix = runtime.get("auth_prefix", "")
        headers[auth_header] = f"{auth_prefix}{runtime['api_key']}".strip()
        extra_headers = runtime.get("request_headers", {})
        if isinstance(extra_headers, dict):
            headers.update(extra_headers)

        req = Request(runtime["endpoint"], data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=300) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"azure_ai HTTP {e.code}: {err_body}") from e
        except URLError as e:
            raise RuntimeError(f"azure_ai connection error: {e}") from e

        data = json.loads(raw)
        if request_format == "anthropic_messages" and isinstance(data, dict):
            content_parts = data.get("content")
            if isinstance(content_parts, list):
                text_parts = []
                for part in content_parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                return {"choices": [{"message": {"content": "".join(text_parts)}}]}

        # Normalize non-OpenAI-like responses to the shape used by current code.
        if isinstance(data, dict) and "choices" in data:
            for c in data.get("choices", []):
                msg = c.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                    parts = []
                    for part in msg["content"]:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    c["message"]["content"] = "".join(parts)
            return data

        text = ""
        if isinstance(data, dict):
            if isinstance(data.get("output_text"), str):
                text = data["output_text"]
            elif isinstance(data.get("content"), str):
                text = data["content"]
            elif isinstance(data.get("response"), str):
                text = data["response"]
        if not text:
            text = str(data)
        return {"choices": [{"message": {"content": text}}]}

    raise RuntimeError(f"Unsupported runtime provider: {runtime.get('provider')}")


def parse_sectioned_prompt(s):
    result = {}
    current_header = None

    for line in s.split("\n"):
        line = line.strip()

        if line.startswith("# "):
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans("", "", string.punctuation))
            result[current_header] = ""
        elif current_header is not None:
            result[current_header] += line + "\n"

    return result


def extract_all_quoted_text(sentence):
    parts = sentence.split('"')

    if len(parts) >= 3:
        quoted_texts = [parts[i] for i in range(1, len(parts), 2)][0]
    else:
        quoted_texts = sentence

    return quoted_texts


def llm_attention(config, inputs, Output, attention_dict, gpt_model, characteristic=""):
    attention_text = ""
    for attention, weight in attention_dict.items():
        attention_prompt = f"""
            output:
            \"{Output}\"

            What is the {attention} of the output in one sentence?
            """
        res = chatGPT(attention_prompt, model=gpt_model, temperature=0.0)
        attention_text += res[0]

    return attention_text


def generate_prompt(config, inputs, output, gradient={}, gpt_model=None, max_tokens=4096, instruction_characteristic=""):
    if gpt_model is None:
        gpt_model = config.get("generator_llm_model", "gpt-4o")

    print("config.theme: ", config["theme"])
    if gradient == {}:
        prompt_gen_template = f"""
                            User_Input:
                            \"{inputs}\"

                            Output:
                            \"{output}\"

                            Your task is to generate an instruction based on the provided User Input and Output. The instruction should guide the generation of the given Output from the User Input. The instruction should be related to the topic of \"{config['theme']}\".

                            The instruction is wrapped with <START> and <END>.
                            """

        res_list = chatGPT(prompt_gen_template, n=1, model=gpt_model, max_tokens=max_tokens, temperature=0.0)
        res = res_list[0] if res_list else None

        if res is None:
            print("[Warning] ChatGPT returned no response.")
            return None

        feedback = utils.parse_tagged_text(res, "<START>", "<END>")
        try:
            assert len(feedback) == 1
        except Exception:
            print("[Warning] Failed to extract a single instruction from LLM output.")
            return None
        prompt = feedback[0]
        return prompt

    attention = llm_attention(config, inputs, output, gradient, gpt_model, instruction_characteristic)

    attention_prompt = f"""
                        User_Input:
                        \"{inputs}\"

                        Output:
                        \"{output}\"

                        Output_characteristic:
                        \"{attention}\"

                        Your task is to generate an instruction based on the provided User_Input and Output. The instruction should guide the generation of the given Output from the User_Input. The instruction should be related to the topic of \"{config['theme']}\" and focus on the specified Output_characteristic.

                        The instruction is wrapped with <START> and <END>.
                        """

    res_list = chatGPT(attention_prompt, n=1, model=gpt_model, temperature=0.0)
    res = res_list[0] if res_list else None

    if res is None:
        print("[Warning] chatGPT returned no response.")
        return None

    feedback = utils.parse_tagged_text(res, "<START>", "<END>")
    try:
        assert len(feedback) == 1
    except Exception:
        print("[Warning] Failed to extract a single instruction from LLM output.")
        return None
    opt_prompt = feedback[0]
    return opt_prompt


def pre_pruning(user_input, prompt, model="gpt-4o"):
    instruction = f"""
    User Input:
    \"{user_input}\"

    Prompt:
    \"{prompt}\"

    I provide a Prompt and User Input. Please identify all parts of the Prompt that are semantically tied to the User Input, and replace them with placeholders "{{}}". Keep the sentence structure intact. Return only the masked prompt.
    The masked prompt is wrapped with <START> and <END>.
    """
    res = chatGPT(instruction, n=1, model=model, temperature=0.0)[0]
    feedback = utils.parse_tagged_text(res, "<START>", "<END>")
    pre_prompt = feedback[0]
    return pre_prompt


def llm_based_evaluation(target_output, generated_output, model="gpt-4o"):
    system_prompt = """
    You are an expert evaluator. The Target Text is the ground truth. The Generated Text should be evaluated against it.
    Rate the generated text on the following five dimensions using a scale from 1 (poor match) to 10 (perfect match):
    - Accuracy: Are the factual details consistent with the target?
    - Completeness: Does it cover all key content from the target?
    - Tone: Is the style and formality consistent with the target?
    - Sentiment: Is the emotional attitude similar?
    - Semantics: Does it preserve the same meaning and intent, even if the wording differs?
    Return only the scores in this exact JSON format (no extra text):

    {
    "Accuracy": X,
    "Completeness": X,
    "Tone": X,
    "Sentiment": X,
    "Semantics": X
    }
    """

    user_prompt = f"""
    Target Text: \"{target_output}\"

    Generated Text: \"{generated_output}\"
    """
    res = chatGPT_inference(system_prompt=system_prompt, text=user_prompt, model=model, temperature=0)[0]
    return res


def chatGPT(
    text,
    temperature=0.7,
    n=1,
    top_p=1,
    stop=None,
    max_tokens=4096,
    presence_penalty=0,
    frequency_penalty=0,
    model="gpt-4o",
    logit_bias={},
):
    messages = [{"role": "user", "content": text}]

    response = None
    retry_count = 0
    max_retries = 1
    while response is None:
        try:
            response = _create_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                n=n,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as e:
            retry_count += 1
            if "This model's maximum context length" in str(e):
                print(e)
                with open("ERR.txt", "a") as outf:
                    outf.write(json.dumps(str(text)) + "\n")
                sys.exit(1)
            if "is greater than the maximum" in str(e):
                raise BatchSizeException()
            if "We could not parse the JSON body of your request" in str(e):
                try:
                    json.dumps({"role": "user", "content": text})
                except Exception as json_err:
                    print("Invalid JSON content in text:", text)
                    print("Serialization error:", json_err)
                else:
                    print("JSON seems valid, but OpenAI still failed.")
                if retry_count >= max_retries:
                    print("Giving up after 1 retries on JSON error.")
                    return []
            print(e)
            print("Retrying......")
            time.sleep(20)
    if response is None:
        return None
    return [choice["message"]["content"] for choice in response["choices"]]


def chatGPT_inference(
    system_prompt,
    text,
    temperature=0.7,
    n=1,
    top_p=1,
    stop=None,
    max_tokens=4096,
    presence_penalty=0,
    frequency_penalty=0,
    model="gpt-4o",
    logit_bias={},
):
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})

    retry_count = 0
    max_retries = 1
    response = None
    while response is None:
        try:
            response = _create_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                n=n,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as e:
            retry_count += 1
            if "This model's maximum context length" in str(e):
                print(e)
                with open("ERR.txt", "a") as outf:
                    outf.write(json.dumps(str(text)) + "\n")
                sys.exit(1)
            if "is greater than the maximum" in str(e):
                raise BatchSizeException()
            if "We could not parse the JSON body of your request" in str(e):
                try:
                    json.dumps({"role": "user", "content": text})
                except Exception as json_err:
                    print("Invalid JSON content in text:", text)
                    print("Serialization error:", json_err)
                else:
                    print("JSON seems valid, but OpenAI still failed.")
                if retry_count >= max_retries:
                    print("Giving up after 1 retries on JSON error.")
                    return []
            print(e)
            print("Retrying......")
            time.sleep(20)
    return [choice["message"]["content"] for choice in response["choices"]]


if __name__ == "__main__":
    pass
