# Packages
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

#CNN Encoder to extract features from image
class CNN_Encoder(nn.Module):
	def __init__(self,embed_size):
		super(CNN_Encoder,self).__init__()
		# Load the pretrained ResNet-152 model and replace the top fc layer
		resnet = models.resnet(pretrained=True)
		modules = list[resnet.children()[:,-1]]
		self.resnet = nn.Sequential(*modules)
		self.linear = nn.Linear(resnet.fc.in_features,embed_size)
		self.bn = nn.BatchNorm2D(embed_size,momentum = 0.01)

	def forward(self,images):
		#Extract feature vectors from input images.
		with torch.no_grad():
			features = self.resnet(images)
		features = features.reshape(features.size(0),-1)
		features = self.linear(features)
		features = self.bn(features)
		return features 

# LSTM decoder trained as a language model conditioned on the 
# feature vector from CNN Encoder
class RNN_Decoder():
	def __init__(self,embed_size,hidden_dim,vocab_size,num_layers,max_seq_length=20):
		super(RNN_Decoder).__init__()
		self.embed = nn.Embedding(vocab_size,embed_size)
		self.lstm = nn.LSTM(embed_size,hidden_dim,num_layers,batch_first=True)
		self.linear = nn.Linear(hidden_size,vocab_size)
		self.max_seq_length = max_seq_length

	def forward(self,features,captions,lengths):
		# Decode the image feature vectors from encoder and generates the caption
		embeddings = self.embed(captions)
		# feature vector:(1,embed_size) is reshaped to (1,1,embed_size)
		# and concatenated to the embeddings of the captions :(1,lengths,embed_size)
		# feature vector act as the initiator or start vector to predict the first word of the 
		# caption.embeddings size after concatenation is :(1,lengths+1,embed_size).
		embeddings = torch.cat((features.unsqueeze(1),embeddings),dim=1)
		packed = pack_padded_sequence(embeddings,lengths,batch_first=True)
		hiddens,_ = self.lstm(packed)
		outputs = self.linear(hiddens)
		return outputs

	def sample(self,features,states = None):
		#Generate captions for given image features using greedy search\
		sampled_ids = []
		inputs = features.unsequeeze(1)
		for i in range(self.max_seq_length):
			hiddens,states = self.lstm(inputs,states)  # hiddens: (batch_size,1,hidden_size)
			outputs = self.linear(hiddens.squeeze(1))  # outputs: (batch_size,vocab_size)
			_,predicted = torch.max(outputs,dim=1)	   # predicted : (batch_size)
			sampled_ids.append(predicted)
			inputs = self.embed(predicted)			   # inputs : (batch_size,embed_size)
			inputs = inputs.unsqueeze(1)			   # inputs : (batch_size,1,embed_size)
		sampled_ids = torch.stack(sampled_ids,1)       # sampled_ids : (batch_size,max_seq_length)
		return sampled_ids




