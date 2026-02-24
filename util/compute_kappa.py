import json

path = 's:/Github/post-intent-detection/data/overlapped_annotations/'
LABELS = ['ADVICE_SEEKING', 'PERSONAL_EXPERIENCE', 'OPINION', 'OTHER']

posts = []
seen_ids = set()
for fname in ['annotated_1.json', 'annotated_2.json', 'annotated_3.json']:
    data = json.load(open(path + fname, 'r', encoding='utf-8'))
    for p in data:
        if p['id'] not in seen_ids:
            posts.append(p)
            seen_ids.add(p['id'])

N = len(posts)
print('Total unique posts:', N)

orig = [p['original_label'] for p in posts]
new  = [p['new_label']      for p in posts]

# Observed agreement
agree = sum(o == n for o, n in zip(orig, new))
obs_agree = agree / N
print('Agreed pairs: ' + str(agree) + ' / ' + str(N))
print('Observed agreement: ' + str(round(obs_agree * 100, 2)) + '%')

# Confusion matrix
print()
print('Confusion Matrix (rows=original_label, cols=new_label):')
header = ''.ljust(22) + '  '.join(l[:8].ljust(8) for l in LABELS)
print(header)
for ol in LABELS:
    row = ol.ljust(22)
    for nl in LABELS:
        cnt = sum(1 for o, n in zip(orig, new) if o == ol and n == nl)
        row += str(cnt).ljust(10)
    print(row)

# Cohen's Kappa
k = len(LABELS)
orig_counts = [sum(1 for o in orig if o == l) for l in LABELS]
new_counts  = [sum(1 for n in new  if n == l) for l in LABELS]

P_e = sum((orig_counts[i] / N) * (new_counts[i] / N) for i in range(k))
kappa = (obs_agree - P_e) / (1 - P_e)

print()
print('Expected agreement by chance (P_e): ' + str(round(P_e, 4)))
print("Cohen's Kappa: " + str(round(kappa, 3)))

if kappa <= 0.20:
    interp = 'Slight agreement'
elif kappa <= 0.40:
    interp = 'Fair agreement'
elif kappa <= 0.60:
    interp = 'Moderate agreement'
elif kappa <= 0.80:
    interp = 'Substantial agreement'
else:
    interp = 'Almost perfect agreement'
print('Interpretation: ' + interp)
