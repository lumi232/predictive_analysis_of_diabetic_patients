"""
Showing how to use the model with some time series data.

NB! This is not a full training loop. You have to write the training loop yourself. 

I.e. this code is just a starting point to show you how to initialize the model and provide its inputs

If you do not know how to train a PyTorch model, it is too soon for you to dive into transformers imo :) 

You're better off starting off with some simpler architectures, e.g. a simple feed forward network, in order to learn the basics
"""

import dataset as ds
import utils
from torch.utils.data import DataLoader
import torch
import datetime
import transformer_timeseries as tst
import numpy as np

# Hyperparams
test_size = 0.1
batch_size = 128
target_col_name = "GlucoseValue"
timestamp_col = "DisplayTime"
# Only use data from this date and onwards
#cutoff_date = datetime.datetime(2017, 1, 1) 

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 48 # length of input given to decoder
enc_seq_len = 168 # length of input given to encoder
output_sequence_length = 48 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False

# Define input variables 
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

input_size = len(input_variables)

# Read data
data = utils.read_data(data_path="data/processed_data.csv", timestamp_col_name=timestamp_col)

# Remove test data from dataset
training_data = data[:-(round(len(data)*test_size))]
test_data = data[-(round(len(data)*test_size)):]

# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
# Should be training data indices only
training_indices = utils.get_indices_entire_sequence(
    data=training_data, 
    window_size=window_size, 
    step_size=step_size)
test_indices = utils.get_indices_entire_sequence(
    data=test_data, 
    window_size=window_size, 
    step_size=step_size)
# Making instance of custom dataset class
training_data = ds.TransformerDataset(
    data=torch.tensor((training_data[input_variables].values).astype(np.float32)).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )

test_data = ds.TransformerDataset(
    data=torch.tensor((test_data[input_variables].values).astype(np.float32)).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )

# Making dataloader
training_data = DataLoader(training_data, batch_size)
test_data = DataLoader(test_data, batch_size)

i, batch = next(enumerate(training_data))

src, trg, trg_y = batch

# Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
if batch_first == False:

    shape_before = src.shape
    src = src.permute(1, 0, 2)
    print("src shape changed from {} to {}".format(shape_before, src.shape))

    shape_before = trg.shape
    trg = trg.permute(1, 0, 2)
    print("src shape changed from {} to {}".format(shape_before, trg.shape))

model = tst.TimeSeriesTransformer(
    input_size=len(input_variables),
    dec_seq_len=enc_seq_len,
    batch_first=batch_first,
    num_predicted_features=1,
    out_seq_len=output_sequence_length
    )

# Make src mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, enc_seq_len]
src_mask = utils.generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=enc_seq_len
    )

# Make tgt mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.generate_square_subsequent_mask( 
    dim1=output_sequence_length,
    dim2=output_sequence_length
    )

output = model(
    src=src,
    tgt=trg,
    src_mask=src_mask,
    tgt_mask=tgt_mask
    )

import torch
import inference
import utils

epochs = 10

forecast_window = window_size #48 # supposing you're forecasting 48 hours ahead

#enc_seq_len = 168 # supposing you want the model to base its forecasts on the previous 7 days of data

optimizer = torch.optim.Adam(params=model.parameters())

criterion = torch.nn.MSELoss()

# Iterate over all epochs
for epoch in range(epochs):

    # Iterate over all (x,y) pairs in training dataloader
    for i, (src, tgt, tgt_y) in enumerate(training_data):

        # zero the parameter gradients
        optimizer.zero_grad()

        # # Generate masks
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=forecast_window
            )

        src_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=enc_seq_len
            )

        # Make forecasts
        prediction = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        # Compute and backprop loss
        loss = criterion(tgt_y, prediction)

        loss.backward()

        # Take optimizer step
        optimizer.step()

        print(epoch)


    # Iterate over all (x,y) pairs in validation dataloader
    model.eval()

    with torch.no_grad():
    
        for i, (src, _, tgt_y) in enumerate(test_data):

            prediction = inference.run_encoder_decoder_inference(
                model=model, 
                src=src, 
                forecast_window=forecast_window,
                batch_size=src.shape[1]
                )

            loss = criterion(tgt_y, prediction)
