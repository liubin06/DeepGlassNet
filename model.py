import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


class DeepGlassNet(nn.Module):
    def __init__(self, num_components, embedding_dim, fm_dim,num_heads, out_dim,dropout_rate=0.1):
        super(DeepGlassNet, self).__init__()
        self.num_components = num_components
        self.embedding_dim = embedding_dim
        self.fm_dim = fm_dim
        self.num_heads = num_heads
        self.out_dim = out_dim

        self.dropout_rate = dropout_rate

        # 1. Each component is parameterized with a learnable embedding vector
        self.component_embeddings = nn.Parameter(torch.randn(num_components,embedding_dim))
        nn.init.xavier_uniform_(self.component_embeddings)

        # 2. A low-rank matrix is used to parameterize the adjacency matrix that governs interactions between components.
        self.v = nn.Parameter(torch.randn(self.num_components, self.fm_dim))
        nn.init.xavier_uniform_(self.v)

        # 3. Self attention
        self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=self.num_heads)

        # 4. Projection head
        self.projector_layer = nn.Sequential(
                                        nn.Linear(self.num_components*self.embedding_dim, 1024, bias =False),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(),
                                        nn.Dropout(p=self.dropout_rate),
                                        nn.Linear(1024, self.out_dim,bias=True)
        )

    def forward(self, component_proportion):
        '''
        :param component_proportion: [bs, num_components]
        :return: feature representations: [bs, out_dim]
        '''
        # 1. Proportion Modulated Embedding Layer
        batch_size = component_proportion.shape[0]
        component_proportion = component_proportion.unsqueeze(-1).repeat(1, 1, self.embedding_dim) #[bs, num_components, embedding_dim]
        fc_1 = component_proportion * self.component_embeddings.unsqueeze(0)      # [bs * num_components, embedding_dim ]


        # 2. Graph Convolution Layer
        norm_v = F.normalize(self.v,dim=-1)
        adj_matrix = torch.matmul(norm_v, norm_v.T)
        # adj_matrix = torch.matmul(self.v, self.v.T)
        adj_matrix1 = adj_matrix.unsqueeze(0).expand(batch_size,-1,-1)      # [bs * num_components * num_components]
        cov_layer = torch.matmul(adj_matrix1,fc_1)                          # [bs * num_components * embedding_dim]
        fc_2 = fc_1 + cov_layer/(self.num_components-1)

        # 3. Self-Attention Layer
        fc_2 = fc_2.permute(1,0,2)                                   # [num_component, bs, embedding_dim]
        fc_3, attention_weight = self.attention_layer(fc_2,fc_2,fc_2)       # [num_component, bs, embedding_dim]
        fc_3 = fc_3.transpose(0,1).contiguous().view(batch_size,-1)        # [bs, num_componen * embedding_dim]

        # 4. Nonlinear Projection Layer
        fc_4 = self.projector_layer(fc_3)

        return F.normalize(fc_4,dim=-1)

