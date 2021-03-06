import numpy as np
import torch
import torch.nn as nn

cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def segment_max(logit, n_seg, seg_i, idx_j):
	max_seg_numel = idx_j.max().item() + 1
	seg_max = logit.new_full((n_seg, max_seg_numel), -np.inf)
	seg_max = seg_max.index_put_((seg_i, idx_j), logit).max(dim=1)[0]
	return seg_max[seg_i]


def segment_sum(logit, n_seg, seg_i):
	norm = logit.new_zeros(n_seg).index_add(0, seg_i, logit)
	return norm[seg_i]


def segment_softmax(logit, n_seg, seg_i, idx_j, temperature):
	logit_max = segment_max(logit, n_seg, seg_i, idx_j).detach()
	logit = torch.exp((logit - logit_max) / temperature)
	logit_norm = segment_sum(logit, n_seg, seg_i)
	prob = logit / (logit_norm + 1e-8)
	return prob


def segment_multihead_expand(seg_i, n_seg, n_head):
	i_head_shift = n_seg * seg_i.new_tensor(torch.arange(n_head))
	seg_i = (seg_i.view(-1, 1) + i_head_shift.view(1, -1)).view(-1)
	return seg_i


class CoAttention(nn.Module):
	def __init__(self,
			node_in_feat,
			node_out_feat,
			n_head=1,
			dropout=0.1):
		super().__init__()
		self.temperature = np.sqrt(node_in_feat)

		self.n_head = n_head
		self.multi_head = self.n_head > 1

		self.key_proj = nn.Linear(node_in_feat, node_in_feat * n_head, bias=False)
		self.val_proj = nn.Linear(node_in_feat, node_in_feat * n_head, bias=False)
		nn.init.xavier_normal_(self.key_proj.weight)
		nn.init.xavier_normal_(self.val_proj.weight)

		self.softmax = nn.Softmax(dim=1)

		self.out_proj = nn.Sequential(
			nn.Linear(node_in_feat * n_head, node_out_feat),
			nn.LeakyReLU())


	def forward(self, x1, masks1, x2, masks2, entropy=[]):

		h1 = self.key_proj(x1)
		h2 = self.key_proj(x2)

		e12 = torch.bmm(h1, torch.transpose(h2, -1, -2))
		e21 = torch.bmm(h2, torch.transpose(h1, -1, -2))

		a12 = self.softmax(e12 + (masks2.unsqueeze(1) - 1.0) * 1e9)
		a21 = self.softmax(e21 + (masks1.unsqueeze(1) - 1.0) * 1e9)

		#a12 = torch.ones_like(e12) * masks2.unsqueeze(1)
		#a21 = torch.ones_like(e21) * masks1.unsqueeze(1)

		n12 = torch.bmm(a12, self.val_proj(x2))
		n21 = torch.bmm(a21, self.val_proj(x1))

		# TODO!! Remove this while training, this is just for entropy.
		#translation = torch.ones_like(translation)

		# Entropy computation
		#ent1 = node1.new_zeros((n_seg1, 1)).index_add(0, seg_i1, torch.sum(node1_edge * torch.log(node1_edge + 1e-7), -1))
		#ent2 = node2.new_zeros((n_seg2, 1)).index_add(0, seg_i2, torch.sum(node2_edge * torch.log(node2_edge + 1e-7), -1))
		#entropy.append(ent1)
		#entropy.append(ent2)

		msg1 = self.out_proj(n12)
		msg2 = self.out_proj(n21)

		#Check if 1 -> 1 and 2 -> 2.
		return msg1, msg2, a12, a21


class MessagePassing(nn.Module):
	def __init__(self,
				 node_in_feat, # probably d_node
				 node_out_feat, # probably d_hid
				 dropout = 0.1):
		super(MessagePassing, self).__init__()

		self.node_proj = nn.Sequential(
			nn.Linear(node_in_feat, node_out_feat, bias=False))

	def compute_adj_mat(self, A):
		batch, N = A.shape[:2]
		A_hat = A
		I = torch.eye(N).unsqueeze(0).to(cuda_device)
		A_hat = A + I
		#D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
		#L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
		return A_hat

	def forward(self, x, A, mask):
		# print('in', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
		if len(A.shape) == 3:
			A = A.unsqueeze(3)

		new_A = self.compute_adj_mat(A[:, :, :, 0]) #only one relation type
		x = self.node_proj(x)
		x = torch.bmm(new_A, x)

		if len(mask.shape) == 2:
			mask = mask.unsqueeze(2)
		x = x * mask  # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc which affects node embeddings in the following layers
		return x


class CoAttentionMessagePassingNetwork(nn.Module):
	def __init__(self,
		hidden_dim, #d_node, # filters
		n_prop_step, #n_prop_step
		n_head=1,
		dropout=0.1,
		update_method='res'):

		super().__init__()

		self.n_prop_step = n_prop_step

		self.update_method = update_method

		self.mps = nn.ModuleList([
			MessagePassing(
				node_in_feat=hidden_dim,
				node_out_feat=hidden_dim, dropout=dropout)
			for step_i in range(n_prop_step)])

		self.coats = nn.ModuleList([
			CoAttention(
				node_in_feat=hidden_dim,
				node_out_feat=hidden_dim, n_head=n_head, dropout=dropout)
			for step_i in range(n_prop_step)])

		self.pre_readout_proj = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.LeakyReLU())

		self.fc = nn.Linear(3*hidden_dim, hidden_dim)


	def forward(
			self,
			x1, A1, masks1,
			x2, A2, masks2,
			entropies=[]):
		for step_i in range(self.n_prop_step):
			#if step_i >= len(entropies):
			#	entropies.append([])
			inner_msg1 = self.mps[step_i](x1, A1, masks1)
			inner_msg2 = self.mps[step_i](x2, A2, masks2)
			#outer_msg1, outer_msg2 = self.coats[step_i](
			#	x1, masks1,
			#	x2, masks2, [])
				# node2, out_seg_i2, out_idx_j2, entropies[step_i])

			if self.update_method == "res":
				x1 = x1 + inner_msg1 #+ outer_msg1
				x2 = x2 + inner_msg2 #+ outer_msg2
			else:
				x1 = self.fc(torch.cat([x1, inner_msg1, outer_msg1], -1))
				x2 = self.fc(torch.cat([x2, inner_msg2, outer_msg2], -1))

		g1_vec = self.readout(x1, masks1)
		g2_vec = self.readout(x2, masks2)

		return g1_vec, g2_vec

	def readout(self, node, mask):
		#print(node.shape)
		node = self.pre_readout_proj(node)
		if len(mask.shape) == 2:
			mask = mask.unsqueeze(2)
		node = node * mask
		mask = mask.squeeze()
		readout = torch.mean(node, 1) / torch.mean(mask, 1, keepdim=True)
		return readout


class GraphGraphInteractionNetwork(nn.Module):
	def __init__(
			self,
			in_features, #d_atom_feat,
			out_features,
			hidden_dim, #d_node, # filters
			n_prop_step, #n_hidden
			n_head=1,
			dropout=0.1,
			update_method='res'):

		super().__init__()

		self.atom_proj = nn.Linear(in_features, hidden_dim)

		self.encoder = CoAttentionMessagePassingNetwork(
			hidden_dim=hidden_dim,
			n_prop_step=n_prop_step, n_head=n_head,
			update_method=update_method, dropout=dropout)
		self.lbl_predict = nn.Linear(hidden_dim, out_features)


	def forward(self, data1, data2, entropies=[]):
		x1, A1, masks1 = data1[:3]
		x2, A2, masks2 = data2[:3]
		#print(x1.shape)

		x1 = self.atom_proj(x1)
		x2 = self.atom_proj(x2)

		d1_vec, d2_vec = self.encoder(x1, A1, masks1,
		 	x2, A2, masks2, entropies)

		pred1 = self.lbl_predict(d1_vec)
		pred2 = self.lbl_predict(d2_vec)
		return pred1, pred2
