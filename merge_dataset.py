import pandas as pd

fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')

fake['label'] = 0  # 0 = Fake
true['label'] = 1  # 1 = Real

df = pd.concat([fake, true], ignore_index=True)
df.to_csv('data/fake_or_real_news.csv', index=False)

print("✅ Dataset merged and saved as data/fake_or_real_news.csv")

