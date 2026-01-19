import pickle

with open('outputs/attention_patterns.pkl', 'rb') as f:
    data = pickle.load(f)

print('Total examples:', len(data))
print('Type of first item:', type(data[0]))

if isinstance(data[0], dict):
    print('\nKeys in first item:', list(data[0].keys()))
    print('\nIgala sentence:', data[0].get('igala_sentence', data[0].get('igala', 'KEY NOT FOUND')))
    print('\nEnglish sentence:', data[0].get('english_sentence', data[0].get('english', 'KEY NOT FOUND')))
