# Individual-Paper-Computer-Science
import numpy as np
import random
import json
import copy as cp
import re

file = json.load(open
                 ('C:\Users\Gebruiker\Desktop\Sara\Econometrie\Master\Blok 2\Computer Science\TVs-all-merged.json', ))
model_words = []
for key, value in file.items():
    for item in value:
        all_model_words = re.finditer("[a-zA-Z0-9]*(([0-9]+[ˆ0-9ˆ,ˆ ]+)|([ˆ0-9ˆ,ˆ ]+[0-9]+))[a-zA-Z0-9]*",
                                      item['title'])
        model_words0 = []
        for model_word in all_model_words:
            model_words0.append(model_word.group())
        model_words.append(model_words0)

universal_set = list(set([item for sublist in model_words for item in sublist]))
boolean_matrix = np.ndarray(shape=(len(universal_set), len(model_words)))
for i in range(len(universal_set)):
    for j in range(len(model_words)):
        if universal_set[i] in model_words[j]:
            boolean_matrix[i, j] = 1
        else:
            boolean_matrix[i, j] = 0

permutations = 500
signature_matrix = np.ndarray(shape=(permutations, boolean_matrix.shape[1]))
permutation = list(range(1, boolean_matrix.shape[0] + 1))
for i in range(permutations):
    random.shuffle(permutation)
    for c in range(boolean_matrix.shape[1]):
        one_values = list(np.nonzero(boolean_matrix[:, c])[0])
        one_value_in_permutation = np.array(permutation)[one_values]
        signature_matrix[i, c] = min(one_value_in_permutation)

t = 0.8 # This can be done for several t
band_values = []
for i in range(1, signature_matrix.shape[0] + 1):
    if signature_matrix.shape[0] % i == 0:
        band_values.append(i)
t_values = []
for band in band_values:
    t_values.append((1 / band) ** (1 / (signature_matrix.shape[0] / band)))
i = np.argmin(np.abs(np.array(t_values) - t))
bands = band_values[i]

signature_matrix_bands = np.array_split(signature_matrix, bands)
candidates = []
for band in signature_matrix_bands:
    band_cols = []
    for c in range(band.shape[1]):
        mod = list(band[:, c])
        band_cols.append(mod)
    for i in range(len(band_cols)):
        for j in range(i + 1, len(band_cols)):
            if band_cols[i] == band_cols[j]:
                candidates.append([i, j])
candidates = [list(x) for x in set(tuple(x) for x in candidates)]

duplicates = np.zeros(shape=(len(model_words), len(model_words)))
duplicate = {}
seen = []
duplicate_pairs = []
for key, value in file.items():
    for item in value:
        if item['modelID'] not in seen:
            seen.append(item['modelID'])
            duplicate[item['modelID']] = 1
        elif item['modelID'] in seen:
            for i in np.where(np.array(seen) == item['modelID'])[0].tolist():
                duplicate_pairs.append([i, len(seen)])
            seen.append(item['modelID'])
            duplicate[item['modelID']] += 1

duplicate_cp = cp.copy(duplicate)
for key, value in duplicate_cp.items():
    if value <= 1:
        del duplicate[key]
duplicates_count = len(duplicate_pairs)

for pair in duplicate_pairs:
    duplicates[pair[0], pair[1]] = 1

pairs = 1317876
correctly_guessed = 0
for pair in candidates:
    if duplicates[pair[0], pair[1]] == 1:
        correctly_guessed += 1

PQ = correctly_guessed / len(candidates)
PC = correctly_guessed / duplicates_count
F1 = (2 * PQ * PQ) / (PQ + PC)
frac_of_comp = len(candidates) / pairs
