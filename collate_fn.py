def collate_fn(batch):
  masked_data_list = []
  for i in batch:
    masked_data_list.append(i)
  
  batched_masked_data = Batch.from_data_list(masked_data_list)

  return batched_masked_data