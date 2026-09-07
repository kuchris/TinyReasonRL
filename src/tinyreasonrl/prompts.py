def make_prompt(numbers, target):
    return (
        "Solve the arithmetic puzzle. Use each supplied number exactly once.\n"
        "Allowed operations: +, -, *, / and parentheses. "
        "Negative and fractional intermediate results are allowed.\n"
        "You may work through the problem before answering.\n"
        "Put only the final expression between <answer> and </answer> tags.\n"
        f"Numbers: {numbers}\nTarget: {target}\nSolution:\n"
    )
