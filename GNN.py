class FoundationalGNN(nn.Module):
  def __init__(self, dataset, hidden_dim=128, num_layers=3, dropout=0.1):
    super(FoundationalGNN, self).__init__()
    self.node_dim = dataset.x.size(1)
    self.edge_dim = dataset.edge_attr.size(1)
    self.hidden_dim = hidden_dim
  
    self.conv_layers = nn.ModuleList()
    self.conv_layers.append(GCNConv(self.node_dim, hidden_dim))

    for _ in range(num_layers - 1):
      self.conv_layers.append(nn.Relu())
      self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
      self.conv_layers.append(nn.Dropout(dropout))
    
    self.dropout = nn.Dropout(dropout)
    self.node_prediction = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, self.node_dim)
    )

    self.edge_prediction = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, self.edge_dim)
    )


    def forward(self):
      x = self.dataset.x
      edge_attr = self.dataset.edge_attr
      edge_index = self.dataset.edge_index

      for layer in self.conv_layers:
        if isinstance(layer, GCNConv):
          x = layer(x, edge_index)
        else:
          x = layer(x)
      
      # Node Prediction
      node_pred = self.node_prediction(x)

      #Edge Prediction: we must first access the embeddings of the nodes that form each edge
      # Then we can concatenate these embeddings to get an edge representation
      edge_source = x[edge_index[0]]
      edge_destination = x[edge_index[1]]
      x = torch.cat([edge_source, edge_destination], dim=-1)

      edge_pred = self.edge_prediction(x)


      return node_pred, edge_pred
