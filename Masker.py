class Masker(torch.utils.data.Dataset):
  def __init__(self, dataset, node_mask_prob=0.15, edge_mask_prob=0.15):
      self.node_mask_prob = node_mask_prob
      self.edge_mask_prob = edge_mask_prob
      self.dataset = dataset

  def __len__(self):
      return len(self.dataset)

  def __getitem__(self, idx):
      data = self.dataset[idx]
      return self._apply_masking(data)

  def _apply_masking(self, data):
    masked_data = copy.deepcopy(data)

    # Masking Node Features
    num_nodes = masked_data.x.size(0)
    num_nodes_to_mask = int(math.ceil((num_nodes * self.node_mask_prob)))
    node_mask_index = random.sample(range(num_nodes), num_nodes_to_mask)
    masked_data.x[node_mask_index, :] = -1


    # Masking Edge Index and Attributes
    num_edges = masked_data.edge_index.size(1)
    num_edges_to_mask = int(math.ceil((num_edges * self.edge_mask_prob)))
    edge_mask_index = random.sample(range(num_edges), num_edges_to_mask)

    columns_to_keep = []
    for i in range(masked_data.edge_index.shape[1]):
      if i != edge_mask_index[0]:
        columns_to_keep.append(i)
    #masked_data.edge_index = data.edge_index[:, columns_to_keep]
    print("Edge Mask Index:", edge_mask_index)
    masked_data.edge_attr[edge_mask_index, :] = -1

    # Original Features/Indices
    orig_node_features = data.x.clone()
    orig_edge_indices = data.edge_index.clone()
    orig_edge_attr = data.edge_attr.clone()
  
    return masked_data
