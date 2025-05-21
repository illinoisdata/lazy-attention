import json

for model in ["blockattnrag", "llmrag", "drag"]:
    with open(f"results/exp1_{model}_2wikimqa.json", "r") as f:
        results = json.load(f)

    ground_truth = results["ground_truths"]
    generated_text = results["generated_texts"]

    ans = 0
    for i in range(len(ground_truth)):
        # print(f"ground_truth: {ground_truth[i]}")
        # print(f"generated_text: {generated_text[i]}")
        # print("-" * 100)
        words = ground_truth[i]
        if any(word in generated_text[i] for word in words):
            ans += 1

    print(f"accuracy for {model} on 2wikimqa: {ans / len(ground_truth)}")