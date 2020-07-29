import pandas as pd
import numpy as np

"""
This script creates the dataset to train the models with.
"""

# JOIN ALL STOCKS ACCORDING TO THEIR DATES
dataset = pd.read_csv('data/original_dataset.csv')
cols = dataset.columns
stock_names = [cols[i].split('_')[0] for i in range(0, len(cols), 2)]
start_date, end_date = dataset[cols[0]].iloc[[19, -1]]  # start from 20th date since there are no SPY before that
dates_range = pd.date_range(start_date, end_date, freq='B')  # all days from start to end date
date_index = [str(i).split(' ')[0] for i in dates_range]  # transform dates into strings
processed_dataset = pd.DataFrame(index=date_index)  # start with blank dataset with all dates
for stock_name in stock_names:  # join stock by stock
    stock_data = dataset[['%s_Date' % stock_name, '%s_Adj Close' % stock_name]].set_index('%s_Date' % stock_name)
    processed_dataset = processed_dataset.join(stock_data)
processed_dataset.columns = stock_names
processed_dataset.index.rename('date', inplace=True)
processed_dataset = processed_dataset.reset_index()  # set index as date column

# AGGREGATE DATASET BY NUMBER OF DAYS
processed_dataset = processed_dataset.fillna(method='ffill').fillna(method='bfill')  # fill forward then fill backward
days_per_block = 5
block_start = 0
stock_names = stock_names[:-1]
rows = []
last_SPY_by_block = []  # will contain last SPY from each n day block
moving_avg_SPY_by_block = []  # will contain moving average of SPY from each n day block
weights = [1 / (2 ** i) for i in range(days_per_block)]  # weights for moving average
while block_start + days_per_block <= len(processed_dataset):  # for each block
    block = processed_dataset[block_start:block_start + days_per_block]
    block_start_date, block_end_date = block['date'].iloc[[0, -1]]
    block_date_range = '%s:%s' % (block_start_date, block_end_date)  # name of block
    row = [block_date_range]  # to be inserted to the dataset
    for i, stock_name in enumerate(stock_names):
        stock_pct_change = block[stock_name].pct_change()  # we take only the pct_change of each stock
        row.extend(list(stock_pct_change)[1:])
    rows.append(row)
    block_spy = list(block['SPY'])
    last_SPY_by_block.append(block_spy[-1])  # append last SPY of block
    moving_avg_SPY_by_block.append(np.average(block_spy, weights=weights))  # append moving average of SPY of block
    block_start += days_per_block

# WRITE THE DATASET TO DISK
cols = ['date range']
for stock_name in stock_names:
    cols.extend(['%s day %d-%d' % (stock_name, i + 1, i + 2) for i in range(days_per_block - 1)])
processed_dataset = pd.DataFrame(rows[:-1], columns=cols)  # skip the last block
label_col_name = 'SPY day %d-%d avg' % (days_per_block + 1, 2 * days_per_block)
label_col = np.array(moving_avg_SPY_by_block[1:]) / np.array(
    last_SPY_by_block[:-1]) - 1  # SPY pct_change between last SPY of block i and moving average of SPY of block i+1
processed_dataset[label_col_name] = label_col
processed_dataset.to_csv('data/processed_dataset.csv', index=False)

# SAVE ACTUAL VALUES OF SPY, TO LATER TRANSFORM THE PREDICTED PCT_CHANGE INTO ACTUAL SPY VALUES
absolute_spy_values = pd.DataFrame({'date range': [row[0] for row in rows[:-1]],
                                    'SPY last of current block': last_SPY_by_block[:-1],
                                    'SPY moving avg of next block': moving_avg_SPY_by_block[1:]})
absolute_spy_values.to_csv('data/absolute_spy_values.csv', index=False)
