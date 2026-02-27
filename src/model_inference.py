import os
from typing import Any


BEDROCK_MODEL_PREFIX = "bedrock:"

OPENAI_MODEL_DROPDOWN_OPTIONS: list[tuple[str, str]] = [
    ("gpt-4.1-mini", "GPT-4.1 Mini (OpenAI)"),
    ("gpt-4", "GPT-4 (OpenAI)"),
    ("gpt-4.1", "GPT-4.1 (OpenAI)"),
    ("gpt-5", "GPT-5 (OpenAI)"),
    ("gpt-5.2-pro", "GPT-5.2-pro (OpenAI)"),
]

AWS_BEDROCK_MODEL_DROPDOWN_OPTIONS: list[tuple[str, str]] = [
    # These are example Bedrock model IDs to expose in the UI.
    # In AWS, ensure each model is enabled for this account/region in Bedrock Model Access.
    # If your org uses different approved models, update this list.
    ("bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0", "Claude 3.7 Sonnet (AWS Bedrock)"),
    ("bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0", "Claude 3.5 Sonnet v2 (AWS Bedrock)"),
    ("bedrock:amazon.nova-pro-v1:0", "Amazon Nova Pro (AWS Bedrock)"),
    ("bedrock:amazon.nova-lite-v1:0", "Amazon Nova Lite (AWS Bedrock)"),
]

MODEL_DROPDOWN_OPTIONS: list[tuple[str, str]] = [
    *OPENAI_MODEL_DROPDOWN_OPTIONS,
    *AWS_BEDROCK_MODEL_DROPDOWN_OPTIONS,
]
SUPPORTED_MODEL_NAMES = {model_name for model_name, _ in MODEL_DROPDOWN_OPTIONS}


def is_bedrock_model(model_name: str) -> bool:
    return (model_name or "").strip().startswith(BEDROCK_MODEL_PREFIX)


def get_bedrock_model_id(model_name: str) -> str:
    raw = (model_name or "").strip()
    if not raw.startswith(BEDROCK_MODEL_PREFIX):
        raise ValueError(f"Model '{raw}' is not an AWS Bedrock model name.")
    model_id = raw[len(BEDROCK_MODEL_PREFIX) :].strip()
    if not model_id:
        raise ValueError("Bedrock model name is missing a model ID.")
    return model_id


def _openai_temperature(model_name: str, temperature: float | None) -> float | None:
    if temperature is None:
        return None
    if (model_name or "").strip().startswith("gpt-5"):
        return None
    return temperature


def _call_openai_model_text(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    require_json: bool,
    temperature: float | None,
    max_tokens: int,
    error_context: str,
) -> str:
    # OpenAI-backed model calls require OPENAI_API_KEY at runtime.
    # In AWS Lambda, set this via environment variables/Secrets Manager/Parameter Store.
    # GovCloud/NIST note (SC-7/SA-9): this path leaves AWS service boundary and may require
    # explicit external system authorization. Prefer `bedrock:*` models when boundary policy requires.
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI model requests.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is required for OpenAI model requests.") from exc

    client = OpenAI(api_key=api_key)
    errors: list[str] = []

    try:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "instructions": system_prompt,
            "input": user_prompt,
        }
        if require_json:
            kwargs["text"] = {"format": {"type": "json_object"}}
        response = client.responses.create(**kwargs)
        raw = getattr(response, "output_text", "") or ""
        if raw.strip():
            return raw
        errors.append("responses API returned empty output")
    except Exception as exc:
        errors.append(f"responses API failed: {exc}")

    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if require_json:
        request_kwargs["response_format"] = {"type": "json_object"}
    openai_temp = _openai_temperature(model_name, temperature)
    if openai_temp is not None:
        request_kwargs["temperature"] = openai_temp

    try:
        response = client.chat.completions.create(**request_kwargs)
        raw = response.choices[0].message.content or ""
        if raw.strip():
            return raw
        errors.append("chat API returned empty output")
    except Exception as exc:
        lower_message = str(exc).lower()
        errors.append(f"chat API failed: {exc}")

        if "temperature" in lower_message and "default" in lower_message:
            try:
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop("temperature", None)
                response = client.chat.completions.create(**retry_kwargs)
                raw = response.choices[0].message.content or ""
                if raw.strip():
                    return raw
                errors.append("chat retry returned empty output")
            except Exception as retry_exc:
                errors.append(f"chat retry failed: {retry_exc}")

        if "not a chat model" in lower_message and "chat/completions" in lower_message:
            try:
                completion_kwargs: dict[str, Any] = {
                    "model": model_name,
                    "prompt": (
                        f"{system_prompt}\n\n"
                        f"{user_prompt}\n\n"
                        f"Return only {'valid JSON' if require_json else 'plain text'}."
                    ),
                    "max_tokens": max_tokens,
                }
                if openai_temp is not None:
                    completion_kwargs["temperature"] = openai_temp
                completion = client.completions.create(**completion_kwargs)
                raw = completion.choices[0].text or ""
                if raw.strip():
                    return raw
                errors.append("completions fallback returned empty output")
            except Exception as fallback_exc:
                errors.append(f"completions fallback failed: {fallback_exc}")

    combined = " | ".join(errors) if errors else "unknown error"
    raise RuntimeError(f"{error_context} failed: {combined}")


def _call_bedrock_model_text(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    require_json: bool,
    temperature: float | None,
    max_tokens: int,
    error_context: str,
) -> str:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("The 'boto3' package is required for AWS Bedrock model requests.") from exc

    model_id = get_bedrock_model_id(model_name)
    # Bedrock region resolution:
    # 1) AWS_BEDROCK_REGION override (recommended when Bedrock region differs from Lambda region)
    # 2) AWS_REGION / AWS_DEFAULT_REGION
    # Ensure the selected model is available in the resolved region.
    # GovCloud note: use `us-gov-west-1` / `us-gov-east-1` and confirm model availability by region.
    region = (
        os.getenv("AWS_BEDROCK_REGION", "").strip()
        or os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
        or None
    )
    try:
        # boto3 uses the standard AWS credential chain (Lambda IAM role, env vars, profile, etc.).
        # Required IAM permissions include Bedrock runtime invocation for the selected models.
        # IA-2/IA-5: in AWS this should be role-based auth, not long-lived static credentials.
        if region:
            client = boto3.client("bedrock-runtime", region_name=region)
        else:
            client = boto3.client("bedrock-runtime")
    except Exception as exc:
        raise RuntimeError(f"{error_context} failed: unable to initialize Bedrock client: {exc}") from exc

    prompt = f"{system_prompt}\n\n{user_prompt}"
    if require_json:
        prompt += "\n\nReturn only valid JSON with no extra prose."

    inference_config: dict[str, Any] = {"maxTokens": max(256, int(max_tokens))}
    if temperature is not None:
        inference_config["temperature"] = float(temperature)

    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=inference_config,
        )
    except Exception as exc:
        raise RuntimeError(f"{error_context} failed: Bedrock request error: {exc}") from exc

    content = response.get("output", {}).get("message", {}).get("content", [])
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
    raw = "".join(parts).strip()
    if not raw:
        raise RuntimeError(f"{error_context} failed: Bedrock model returned empty output.")
    return raw


def call_model_text(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    require_json: bool = True,
    temperature: float | None = 0.0,
    max_tokens: int = 1200,
    error_context: str = "Model request",
) -> str:
    selected_model = (model_name or "").strip()
    if not selected_model:
        raise RuntimeError(f"{error_context} failed: model_name is required.")
    if is_bedrock_model(selected_model):
        return _call_bedrock_model_text(
            model_name=selected_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            require_json=require_json,
            temperature=temperature,
            max_tokens=max_tokens,
            error_context=error_context,
        )
    return _call_openai_model_text(
        model_name=selected_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        require_json=require_json,
        temperature=temperature,
        max_tokens=max_tokens,
        error_context=error_context,
    )
