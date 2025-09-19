from longbench import load_dataset, split_context
from blockbench import BlockBenchArgs, load_dataset

load_dataset('2wiki')

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
# print("Loading dataset...")
# dataset = load_dataset('2wikimqa')
# print(len(dataset.rows))
# for i in range(5):
#     print("-" * 20)
#     context_example = dataset.rows[i].context
#     question_example = dataset.rows[i].input
#     for con in context_example:
#         print("Example context:", con)
#     print("-" * 20)
#     print("Example question:", question_example)
# musique
# samsum
# multi_news

# block attention
# triviaqa
# narrativeqa
# hotpotqa