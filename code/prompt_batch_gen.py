import pandas as pd

# 读取数据

tableA = pd.read_csv('./tableA.csv')
tableB = pd.read_csv('./tableB.csv')
test = pd.read_csv('./test.csv')

# 合并数据，获得所有列
merged = test.merge(tableA, left_on='ltable_id', right_on='id', how='left')
merged = merged.rename(columns=lambda x: x + '_A' if x in tableA.columns and x != 'id' else x)
merged = merged.drop(columns=['id'])
merged = merged.merge(tableB, left_on='rtable_id', right_on='id', how='left')
merged = merged.rename(columns=lambda x: x + '_B' if x in tableB.columns and x != 'id' else x)
merged = merged.drop(columns=['id'])

def format_row(row):
    return ", ".join([f"{col}: {row[col]}" for col in row.index])

prompts = []
for i in range(0, len(merged), 20):
    batch = merged.iloc[i:i+20]
    prompt = "\n".join([format_row(row) for _, row in batch.iterrows()])
    prompts.append(prompt)

with open("prompt_batches.txt", "w", encoding="utf-8") as f:
    for idx, prompt in enumerate(prompts):
        f.write(f"Batch {idx+1}:\n{prompt}\n{'='*40}\n")

print("Saved: prompt_batches.txt")
