def make_prompt(numbers, target):
    return (
        "Solve the arithmetic puzzle. Use each supplied number exactly once.\n"
        "Allowed operations: +, -, *, / and parentheses. "
        "Negative and fractional intermediate results are allowed.\n"
        "You may work through the problem before answering.\n"
        "Put only the final expression between <answer> and </answer> tags.\n"
        f"Numbers: {numbers}\nTarget: {target}\nSolution:\n"
    )


def render_prompt(prompt, tokenizer, config):
    if config.get("prompt_format", "plain") == "plain":
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=config["enable_thinking"])
