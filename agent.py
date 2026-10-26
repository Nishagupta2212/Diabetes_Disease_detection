"""AI assistant (Groq-hosted LLM) that answers questions using the diabetes model as tools."""
import json
import logging
import os

import groq

logger = logging.getLogger(__name__)

MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are the assistant inside "Diabetes Risk Check", an educational web app \
built on a machine-learning model trained on the PIMA Indians Diabetes dataset (768 adult \
female patients of Pima heritage, 8 routine measurements each).

Use the tools to answer: run predict_diabetes_risk for any risk estimate, and look numbers up \
with the dataset and model tools rather than quoting them from memory. For general health \
questions (what a reading means, normal ranges, symptoms, risk factors, prevention, how the \
dataset or model works), call search_health_guidelines first and base your answer on the \
passages it returns, naming the source in brief, e.g. "(ADA)" or "(WHO)". If the passages don't \
cover the question, say so rather than guessing. If the user's question \
needs measurements they have not given, ask for them, or say which values you assumed. Unknown \
lab values can be passed as 0; the model fills them with the dataset median.

This is an educational tool, not a diagnosis. Be clear about the model's limits (about 77% \
accuracy, trained on one specific population) when it matters to the answer, and suggest seeing \
a healthcare professional for real medical decisions. Do not give medication or dosing advice.

Keep replies short and plain: a few sentences or a short list. You may use **bold** and \
"- " bullet lists; no tables or headings."""

FEATURE_DESCRIPTIONS = {
    "Pregnancies": "Number of times pregnant.",
    "Glucose": "Plasma glucose (mg/dL) 2 hours into an oral glucose tolerance test; 0 if unknown.",
    "BloodPressure": "Diastolic blood pressure (mm Hg); 0 if unknown.",
    "SkinThickness": "Triceps skin fold thickness (mm); 0 if unknown.",
    "Insulin": "2-hour serum insulin (µU/mL); 0 if unknown.",
    "BMI": "Body mass index (kg/m²); 0 if unknown.",
    "DiabetesPedigreeFunction": "Family-history score, typically 0.08-2.4 (dataset median 0.37).",
    "Age": "Age in years.",
}

FEATURE_NAMES = list(FEATURE_DESCRIPTIONS)


def function_tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    function_tool(
        "predict_diabetes_risk",
        "Run the app's trained classifier on one patient's eight measurements and return the "
        "estimated probability of diabetes, the risk band, which values were filled in as "
        "unknown, and how each measurement compares with typical adult ranges.",
        {name: {"type": "number", "description": desc} for name, desc in FEATURE_DESCRIPTIONS.items()},
        FEATURE_NAMES,
    ),
    function_tool(
        "get_dataset_statistics",
        "Summary statistics for one column of the training dataset: count, mean, standard "
        "deviation, min, quartiles and max, plus the mean for diabetic and non-diabetic "
        "patients. Use feature 'Outcome' for the overall diabetic share.",
        {"feature": {"type": "string", "enum": FEATURE_NAMES + ["Outcome"]}},
        ["feature"],
    ),
    function_tool(
        "compare_to_dataset",
        "Place one measurement within the training dataset: the percentile among all "
        "patients, among diabetic patients and among non-diabetic patients, and the share "
        "of patients at or above that value who are diabetic.",
        {"feature": {"type": "string", "enum": FEATURE_NAMES}, "value": {"type": "number"}},
        ["feature", "value"],
    ),
    function_tool(
        "search_health_guidelines",
        "Search the app's library of trusted diabetes guidance (WHO, ADA, CDC, AHA) and documentation "
        "about this app's dataset and model. Returns the most relevant passages with their source. "
        "Use short keyword-rich queries, e.g. 'normal 2-hour OGTT glucose range' or 'prevent type 2 "
        "diabetes weight loss'.",
        {"query": {"type": "string", "description": "What to look up."}},
        ["query"],
    ),
    function_tool(
        "get_model_info",
        "The algorithm, cross-validated and held-out accuracy, number of training patients, "
        "and each feature's weight in the model (how strongly it pushes risk up or down).",
        {},
        [],
    ),
]


class AgentUnavailable(Exception):
    """Raised when the assistant cannot run, with a message safe to show the user."""


_client = None


def get_client():
    global _client
    if _client is None:
        # Strip stray spaces or newlines that often come along when a key is pasted.
        api_key = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            raise AgentUnavailable(
                "The AI assistant is not configured. Set the GROQ_API_KEY environment variable."
            )
        _client = groq.Groq(api_key=api_key)
    return _client


def run_tool_call(tool_call, execute_tool):
    """Returns (content for the model, result dict or None on error)."""
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        result = execute_tool(tool_call.function.name, arguments)
        return json.dumps(result), result
    except json.JSONDecodeError:
        return "Error: tool arguments were not valid JSON.", None
    except ValueError as exc:
        return f"Error: {exc}", None


def chat(history, execute_tool):
    """Run one assistant turn.

    history: list of {"role": "user"|"assistant", "content": str}, ending with a user message.
    execute_tool: callable(name, input_dict) -> JSON-serialisable result; raises ValueError
        for bad input.
    Returns (reply_text, tools_used, sources), where sources lists the knowledge passages
    retrieved by search_health_guidelines as {"title", "source", "url"}.
    """
    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    tools_used = []
    sources = []

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                reasoning_format="hidden",
                reasoning_effort="low",
                max_completion_tokens=4096,
            )
        except groq.GroqError as exc:
            # Logged so the real cause shows up in the server (Vercel) logs.
            logger.error("Groq request failed: %r", exc)
            if isinstance(exc, groq.AuthenticationError):
                raise AgentUnavailable("The AI assistant's API key is invalid.") from exc
            if isinstance(exc, groq.RateLimitError):
                raise AgentUnavailable("The AI assistant is busy right now. Please try again in a minute.") from exc
            if isinstance(exc, groq.BadRequestError):
                # Includes "tool_use_failed" when the model writes a malformed tool call.
                raise AgentUnavailable("The AI assistant couldn't handle that request. Try rephrasing it.") from exc
            if isinstance(exc, groq.NotFoundError):
                raise AgentUnavailable(f"The AI model '{MODEL}' is not available on this Groq account.") from exc
            if isinstance(exc, groq.APIStatusError):
                raise AgentUnavailable(
                    f"The AI service returned an error ({exc.status_code}). Please try again."
                ) from exc
            raise AgentUnavailable("The AI assistant could not be reached. Please try again.") from exc

        choice = response.choices[0]
        message = choice.message

        if not message.tool_calls:
            text = (message.content or "").strip()
            if choice.finish_reason == "length":
                text += "\n\n(Reply cut short.)"
            return text or "Sorry, I couldn't come up with an answer.", tools_used, sources

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })
        for tc in message.tool_calls:
            tools_used.append(tc.function.name)
            content, result = run_tool_call(tc, execute_tool)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            if tc.function.name == "search_health_guidelines" and result:
                for p in result.get("passages", []):
                    source = {k: p[k] for k in ("title", "source", "url")}
                    if source not in sources:
                        sources.append(source)

    return "Sorry, that took too many steps. Please try a simpler question.", tools_used, sources
