Code that builds a GNN that can be pre-trained for molecular representation via node feature and edge feature masking. This was built using PyTorch Geometric.

The Masker class also includes logic that masks edge indices, but this logic is not carried forward to the model or to the loss calculations. When looking to build a GNN for pre-training, I had found that it was difficult to find lots of code examples online, especially with respect to masking strategies. Although there are plenty of research papers around the topic, I sometimes had a hard time finding the code that they used. I hope that this code, in particular the masking strategies employed here, are useful to those looking to pre-train a GNN regardless of the field of work. 

I would note that the Loss.py script is currently just a generic loss calculation between the predicted tensor and the true tensor. I need to make additional changes to ensure that loss is calculated between the batched masked tensor and the batched unmasked tensor -- masked and predicted vs orig and unmasked. 

The Trainer.py script is also a generic training script. I need to do more work to ensure that I am properly defining the true vs pred tensors. Since the GNN outputs both a node_feature tensor and an edge_feature tensor, I will not be using the input batched tensor in my loss calculation. So I need to include the orig unmasked data in (probably?) a separate data_loader to loop through when training. 

More work is definitely needed. For those of you who find this even slightly helpful, I would appreciate your feedback. 
