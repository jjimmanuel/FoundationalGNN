class MolecularGNNTrainer():
  def __init__(self, model, optimizer, node_loss, edge_loss, data_loader, device):
    self.model = model
    self.optimizer = optimizer
    self.node_loss = node_loss
    self.edge_loss = edge_loss
    self.data_loader = data_loader
    self.device = device
  
  def epoch(self):
    self.model.train()
    total_loss = 0

    for batch in self.data_loader:
      batch = batch.to(self.device)
      self.optimizer.zero_grad()

      node_pred, edge_pred = self.model(batch)

      node_loss = self.node_loss.feature_loss()
      edge_loss = self.edge_loss.feature_loss()

      loss = node_loss + edge_loss
      loss.backward()
      self.optimizer.step()

      total_loss += loss

    return total_loss / len(self.data_loader)

  def train(self, epochs):
    for epoch in range(1, epochs + 1):
        epoch_loss = self.train_epoch()
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")