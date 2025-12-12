import pandas as pd
import json

# Load the dataset
file_path = 'data/gold_standard_clean.json'

with open(file_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

def calculate_metrics(dataframe):
    # Avoid division by zero if a split is empty
    if dataframe.empty:
        return [0, 0, 0, 0, "0.0"]

    # 1. Number of Unique RCT reports (using pmcid)
    n_rct = dataframe['pmcid'].nunique()
    
    # 2. Number of ICO-triplets (total rows)
    n_ico = len(dataframe)
    
    # 3. Number of binary outcomes
    n_binary = len(dataframe[dataframe['outcome_type'] == 'binary'])
    
    # 4. Number of continuous outcomes
    n_continuous = len(dataframe[dataframe['outcome_type'] == 'continuous'])
    
    # 5. % with data in figure or image table
    has_graphic_data = dataframe[
        (dataframe['is_data_in_figure_graphics'] == True) | 
        (dataframe['is_table_in_graphic_format'] == True)
    ]
    pct_graphic = (len(has_graphic_data) / n_ico * 100)
    
    return [n_rct, n_ico, n_binary, n_continuous, f"{pct_graphic:.1f}"]

# Calculate columns for each split
dev_metrics = calculate_metrics(df[df['split'] == 'DEV'])
test_metrics = calculate_metrics(df[df['split'] == 'TEST'])
few_shot_metrics = calculate_metrics(df[df['split'] == 'FEW-SHOT'])

# Calculate Total (sum of all three splits)
# Note: We use the full dataframe filtered to these 3 splits to ensure Total represents the table contents
df_total = df[df['split'].isin(['DEV', 'TEST', 'FEW-SHOT'])]
total_metrics = calculate_metrics(df_total)

# Create the summary DataFrame
summary_table = pd.DataFrame({
    'Variable / Measure': [
        'No. RCT reports',
        'No. ICO-triplets',
        'No. binary outcomes',
        'No. continuous outcomes',
        '% with data in figure or image table'
    ],
    'Dev': dev_metrics,
    'Test': test_metrics,
    'Few-Shot': few_shot_metrics,
    'Total': total_metrics
})

# Display the table
print(summary_table.to_latex(index=False, caption="Summary of dataset characteristics", label="tab:dataset_summary"))