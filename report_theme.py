"""One palette and one presentation mapping for experiment reports."""

PALETTE = {
    "paper": "#ffffff",
    "canvas": "#f4f6f9",
    "ink": "#172438",
    "muted": "#53647a",
    "line": "#dce3eb",
    "blue": "#245ac7",
    "teal": "#087b78",
    "purple": "#8748b7",
    "orange": "#b45a12",
    "rose": "#b83972",
    "olive": "#647a18",
    "green": "#1a9a6c",
    "magenta": "#e87ba4",
    "indigo": "#4a49a8",
    "brown": "#865132",
    "cyan": "#087aab",
    "good": "#16644d",
    "bad": "#a33446",
}

MODELS = {
    "ministral-base": {"label": "Ministral · Base", "color": "blue"},
    "ministral-instruct": {"label": "Ministral · Instruct", "color": "teal"},
    "qwen-base": {"label": "Qwen3 · Base", "color": "purple"},
    "qwen-instruct": {"label": "Qwen3 · Post-trained", "color": "orange"},
    "deepseek-distill": {"label": "DeepSeek R1 · Distill 1.5B", "color": "rose"},
    "deepscaler": {"label": "DeepScaleR · RL 1.5B", "color": "olive"},
    "phi-reasoning": {"label": "Phi-4-mini · Reasoning", "color": "indigo"},
    "qwen-thinking": {"label": "Qwen3 · Thinking 2507", "color": "brown"},
    "qwen35-reasoning": {"label": "Qwen3.5 · Reasoning 4B", "color": "cyan"},
}
