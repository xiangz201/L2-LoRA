import numpy
import os
import torch
import string
import re

filename = 'results_llama2/hellaswag/cases_arc_challenge'
all_files = os.listdir(filename)
all_files = sorted(
    all_files,
    key=lambda x: int(re.search(r'knowledge_(\d+)', x).group(1)),
    reverse=False
)
npz_files = [f for f in all_files if f.endswith('.npz') and 'attn' not in f and 'mlp' not in f]
all_data = []
print(len(npz_files))
all_low_score = []
all_high_score = []
for i in npz_files:
    filepath = os.path.join(filename, i)
    numpy_result = numpy.load(filepath, allow_pickle=True)
    answer = numpy_result["answer"]
    differences = numpy_result['scores']
    differences = differences.astype(numpy.float32)
    if numpy.isnan(differences).any():
        continue
    low_score = numpy_result['low_score']
    low_score = low_score.astype(numpy.float32)
    all_low_score.append(low_score)
    high_score = numpy_result['high_score']
    all_high_score.append(high_score)
    input_tokens = numpy_result['input_tokens']
    if high_score<=low_score:
        continue
    subject_range = numpy_result['subject_range']
    noise_token = input_tokens[subject_range[0]:subject_range[1]]
    aie = differences - low_score
    aie = numpy.array(aie)
    aie = numpy.sum(aie, axis=0)
    all_data.append(aie)
all_data = numpy.array(all_data)
print(all_data.shape)
all_data = numpy.mean(all_data, axis=0)
all_data = all_data/numpy.sum(all_data)
print("generate layer-level localization results for files:", filename)
print("Normalized Average Indirect Effect:\n", all_data.tolist())
