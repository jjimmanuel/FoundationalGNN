class TotalFeatureLoss():
  def __init__(self, loss_fn=nn.MSELoss()):
    self.loss_fn = loss_fn

  def feature_loss(self, pred, true):
    total_loss = nn.MSELoss()(pred, true)
    return node_loss
