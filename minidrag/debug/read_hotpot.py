import json

# Read the json file
with open("../Hotpot-Bench/hotpot_test_v1.json", "r") as f:
    data = json.load(f)
# Print the first 5 items
num_docs = 0
all_titles = set()
for i in range(len(data)):
    titles = [title_and_sent[0] for title_and_sent in data[i]['context']]
    num_docs += len(titles)
    all_titles.update(titles)
    
print(f"Number of documents: {num_docs}")
print(f"Number of unique titles: {len(all_titles)}")