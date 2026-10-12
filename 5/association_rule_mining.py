import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Import the dataset as DataFrame
df = pd.read_csv('retail_dataset_fixed.csv', header=None)

# Assuming the dataset is in transaction format, where each row is a transaction
# Convert to list of lists, splitting by comma
transactions = []
for row in df.itertuples():
    items_str = str(row[1])
    if items_str.startswith('"') and items_str.endswith('"'):
        items_str = items_str[1:-1]
    items = items_str.split(',')
    transactions.append(items)

# Let's try a different approach: read the file line by line
transactions = []
with open('retail_dataset_fixed.csv', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) > 1:
            items = parts[1:]
            if len(items) == 1 and items[0].startswith('"') and items[0].endswith('"'):
                items = items[0][1:-1].split(',')
            transactions.append(items)

print("Transactions:")
for i, t in enumerate(transactions):
    print(f"{i}: {t}")

# Step 2: Install mlxtend library (already done via pip install)

# Step 3: Using the Apriori algorithm generate a list of items frequently bought together
# First, encode the transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

print("Frequent Itemsets:")
print(frequent_itemsets)

# Step 4: Generate the association rules for the given items from the Apriori algorithm
if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    print("\nAssociation Rules:")
    print(rules)
else:
    print("\nNo frequent itemsets found with the given min_support.")
