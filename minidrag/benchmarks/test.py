from longbench import load_dataset, split_context

# dataset = load_dataset('2wikimqa')
# num_rows = len(dataset.rows)
# num_passages_list = []
# average_length_list = []
# max_length_list = []
# min_length_list = []
# for i in range(num_rows):
#     row = dataset.rows[i]
#     passages = (split_context(row.context))
#     len_passages = len(passages)
#     num_passages_list.append(len_passages)
#     passage_length = [len(p) for p in passages]
#     average_length = sum(passage_length) / len_passages if len_passages > 0 else 0
#     max_length = max(passage_length) if len_passages > 0 else 0
#     min_length = min(passage_length) if len_passages > 0 else 0
#     average_length_list.append(average_length)
#     max_length_list.append(max_length)
#     min_length_list.append(min_length)
    
# print("Average number of passages:", sum(num_passages_list) / num_rows)
# print("Average length of passages:", sum(average_length_list) / num_rows)
# print("Max length of passages:", sum(max_length_list) / num_rows)
# print("Min length of passages:", sum(min_length_list) / num_rows)
print("Loading dataset...")
dataset = load_dataset('2wikimqa')
print(len(dataset.rows))
passage_titles = []
for i in range(len(dataset.rows)):
    passage_title = []
    # print('-' * 20)
    context_example = dataset.rows[i].context
    for passage in context_example:
        title = passage.split("\n")[1]
        passage_title.append(title)
    passage_titles.append(passage_title)
    question_example = dataset.rows[i].input

nums = []
max_overlap = 0
for i in range(len(passage_titles)):
    for j in range(i+1, len(passage_titles)):
        t_i = passage_titles[i]
        t_j = passage_titles[j]
        len_overlap = len(set(t_i) & set(t_j))
        max_overlap = max(max_overlap, len_overlap)
        print(f"Passage {i} and Passage {j} have {len_overlap} overlapping titles.")
        nums.append(len_overlap)
print("Average overlap:", sum(nums) / len(nums))
# counting the frequency of each overlap
from collections import Counter
overlap_counter = Counter(nums)
print("Overlap frequency:", overlap_counter)

print("Max overlap:", max_overlap)
for i in [15, 50]: # [198, 199]:
    print(len(passage_titles[i]), passage_titles[i])
    print(dataset.rows[i].input)
# print(len(passage_titles[87]), passage_titles[87])
# print(dataset.rows[87].input)

# musique
# samsum
# multi_news

# block attention
# triviaqa
# narrativeqa
# hotpotqa