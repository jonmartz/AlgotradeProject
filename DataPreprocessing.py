import pandas as pd
import csv

# join all stocks according to their dates
dataset = pd.read_csv('data/dataset_original.csv')
cols = dataset.columns
stock_names = [cols[i].split('_')[0] for i in range(0, len(cols), 2)]
start_date, end_date = dataset[cols[0]].iloc[[19, -1]]  # start from 20th date since there are no SPY before that
dates_range = pd.date_range(start_date, end_date, freq='B')
date_index = [str(i).split(' ')[0] for i in dates_range]  # transform dates into strings
processed_dataset = pd.DataFrame(index=date_index)  # start with blank dataset with all dates
for stock_name in stock_names:  # join stock by stock
    stock_data = dataset[['%s_Date' % stock_name, '%s_Adj Close' % stock_name]].set_index('%s_Date' % stock_name)
    processed_dataset = processed_dataset.join(stock_data)
processed_dataset.columns = stock_names
processed_dataset.index.rename('date', inplace=True)
processed_dataset = processed_dataset.reset_index()  # set index as date column

# aggregate dataset by number of days
processed_dataset = processed_dataset.fillna(method='ffill').fillna(method='bfill')  # fill forward then fill backward
days_per_block = 5
block_start = 0
stock_names = stock_names[:-1]
x = []
y = []
while block_start + days_per_block <= len(processed_dataset):
    block = processed_dataset[block_start:block_start + days_per_block]
    block_start_date, block_end_date = block['date'].iloc[[0, -1]]
    block_date_range = '%s:%s' % (block_start_date, block_end_date)
    label = block['SPY'].mean()
    for i, stock_name in enumerate(stock_names):
        one_hot = [0] * len(stock_names)
        one_hot[i] = 1
        stock_values = list(block[stock_name])
        x.append([block_date_range] + one_hot + stock_values)
        y.append(label)
    block_start += days_per_block
cols = ['date range'] + stock_names + ['day %d' % (i + 1) for i in range(days_per_block)]
final_dataset = pd.DataFrame(x[:-len(stock_names)], columns=cols)
final_dataset['SPY'] = y[len(stock_names):]
final_dataset.to_csv('data/processed_dataset.csv', index=False)
