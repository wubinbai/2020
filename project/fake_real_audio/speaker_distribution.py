import json
with open('../test_dict_bitrates.json','r') as f:
    dict_bitrates = json.load(f)

df = pdrc('../new_df.csv')

speaker_256k = []
spearker_768k = []
for a, b in dict_bitrates.items():
    if b == ' 256k':
        speaker = df[df['bitrates']==b]
        speaker_256k = speaker['Speaker_ID'].unique()
    if b == ' 768k':
        speaker = df[df['bitrates']==b]
        speaker_768k = speaker['Speaker_ID'].unique()


