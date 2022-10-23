import torch

def naive_random_sampler(lables, sample_n=50):
    # assert:
    #   1) no padding - all sequences in this batch are in the same length
    #   2) possible to sample the anchor itself
    #   3) most of the samples are negative of the corresponding position (anchor)

    batch_size, max_seq_len = labels.shape
    sample_shape = (batch_size, max_seq_len, sample_n)
    sample_row_indices = torch.randint(batch_size, sample_shape, dtype=torch.long)
    sample_col_indices = torch.randint(max_seq_len, sample_shape, dtype=torch.long)

    return sample_row_indices, sample_col_indices

def negative_random_sampler(labels, sample_n=50):
    # assert:
    #   1) no padding - all sequences in this batch are in the same length
    #   2) all samples are negative of the corresponding position (anchor)

    batch_size, max_seq_len = labels.shape
    # each row has different sampling range (except the row it locates)
    # row_id -> sample from [0, bs-2] -> change by (x+row_id+1)%bs -> get [row_id+1, bs-1] & [0, row_id-1]
    # e.g. if row_id=1 & bs=160, we have 1 -> sample from [0, 158] -> chenge to 2 & 0 -> get [2, 159] & [0, 0]
    sample_shape_perrow = (1, max_seq_len, sample_n)
    sample_row_indices = [(torch.randint(batch_size-1, sample_shape_perrow, dtype=torch.long)+row_id+1)%batch_size \
                            for row_id in range(batch_size)]
    sample_row_indices = torch.cat(sample_row_indices) # batch_size * (1, max_seq_len, sample_n) -> (batch_size, max_seq_len, sample_n)
    # col sampling range is not constraint
    sample_shape_allcol = (batch_size, max_seq_len, sample_n)
    sample_col_indices = torch.randint(max_seq_len, sample_shape_allcol, dtype=torch.long)

    return sample_row_indices, sample_col_indices

def postive_random_sampler(labels, window_size=5, sample_n=1):
    # assert:
    #   1) no padding - all sequences in this batch are in the same length
    #   2) all samples are positive of the corresponding position (anchor) 
    #   3) if sample lies in 'out of range' area, weak positive samples are chosed within the same sequence randomly, 
    #      otherwise the strong positive samples within the window of the same sequence

    assert window_size % 2 == 1
    half_window_size = window_size // 2
    batch_size, max_seq_len = labels.shape
    # each row has different sampling range (must be the row it locates)
    sample_shape_perrow = (1, max_seq_len, sample_n)
    sample_row_indices = [torch.full(sample_shape_perrow, row_id, dtype=torch.long) \
                            for row_id in range(batch_size)]
    sample_row_indices = torch.cat(sample_row_indices, 0) # batch_size * (1, max_seq_len, sample_n) -> (batch_size, max_seq_len, sample_n)
    # each col has different sampling range (must be [l, r) and != col_id) where l = min(0, col_id - window_size), r = max(col_id + window_size + 1, max_seq_len)
    # col_id -> sample from [0, window_size-2] -> change by (x+half_window_size+1)%window_size - half_window_size + col_id
    # e.g. if col_id=1 & window_size=5, we have 1 -> sample from [0, 3] -> change to [3, 6] -> [3, 4] & [0, 1] -> [1, 2] & [-2,-1]
    sample_shape_percol = (batch_size, 1, sample_n)
    sample_col_indices = [(torch.randint(window_size-1, sample_shape_percol, dtype=torch.long)+half_window_size+1)%window_size - half_window_size + col_id \
                            for col_id in range(max_seq_len)]
    sample_col_indices = torch.cat(sample_col_indices, 1) # max_seq_len * (batch_size, 1, sample_n) -> (batch_size, max_seq_len, sample_n)
    # avoid out of range
    backup_sample_col_indices = [(torch.randint(max_seq_len-1, sample_shape_percol, dtype=torch.long)+col_id+1)%max_seq_len \
                                    for col_id in range(max_seq_len)]
    backup_sample_col_indices = torch.cat(backup_sample_col_indices, 1) # max_seq_len * (batch_size, 1, sample_n) -> (batch_size, max_seq_len, sample_n)
    sample_col_indices = torch.where(sample_col_indices<0, backup_sample_col_indices, sample_col_indices)
    sample_col_indices = torch.where(sample_col_indices>=max_seq_len, backup_sample_col_indices, sample_col_indices)

    return sample_row_indices, sample_col_indices




