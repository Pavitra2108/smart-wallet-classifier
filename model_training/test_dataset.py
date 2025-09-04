import pandas as pd

# Load your dataset
df = pd.read_csv('Transaction_Dataset.csv')

print(f"Total transactions: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== CATEGORIES IN YOUR DATA ===")
categories = df['Category'].value_counts()
print(categories)

print("\n=== SAMPLE TRANSACTIONS ===")
for category in categories.index[:3]:  # Show first 3 categories
    print(f"\n{category} example:")
    sample = df[df['Category'] == category].iloc[0]
    print(f"  Amount: ${sample['Amount']}")
    print(f"  Merchant: {sample['Merchant']}")
    print(f"  Description: {sample['Description']}")

print("\nYour data looks good! Ready for next steps.")