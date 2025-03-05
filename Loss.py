class NodeFeatureLoss():
  def __init__(self, loss_fn=nn.MSELoss()):
    self.loss_fn = loss_fn

  def feature_loss(self, node_pred, node_true):
    node_loss = nn.MSELoss()(node_pred, node_true)
    return node_loss


class EdgeFeatureLoss():
  def __init__(self, loss_fn=nn.MSELoss()):
    self.loss_fn = loss_fn

  def feature_loss(self, edge_pred, edge_true):
    edge_loss = nn.MSELoss()(edge_pred, edge_true)
    return edge_loss


class TotalFeatureLoss():
  def __init__(self, edge_loss, node_loss):
    self.edge_loss = edge_loss
    self.node_loss = node_loss
  
  def total_loss(self):
    return self.edge_loss + self.node_loss