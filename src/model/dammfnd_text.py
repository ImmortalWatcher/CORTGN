import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import tqdm

# 内部模块导入
from .layers import *
from .pivot import *

# 工具模块导入 (根据 src/model 目录结构向上查找 utils)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_dataloader import data2gpu
from utils.utils import metricsTrueFalse, Recorder, Averager

class DomainAwareTransformer(nn.Module):
    """领域感知 Transformer"""
    def __init__(self, dim=320, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.global_proj = nn.Linear(dim, dim)
        self.weight_proj = nn.Linear(dim, 3)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_reps, domain_rep):
        batch_size = domain_rep.shape[0]
        global_rep = torch.mean(torch.stack(modality_reps, dim=1), dim=1)
        global_rep = self.global_proj(global_rep)
        
        q = self.q_proj(domain_rep + global_rep).view(batch_size, self.num_heads, self.head_dim)
        
        k = torch.stack([self.k_proj(rep) for rep in modality_reps], dim=1)
        v = torch.stack([self.v_proj(rep) for rep in modality_reps], dim=1)
        
        k = k.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).squeeze(2)
        out = out.reshape(batch_size, -1)

        domain_rep = domain_rep + self.dropout(out)
        domain_rep = self.norm1(domain_rep)
        
        ffn_output = self.ffn(domain_rep)
        domain_rep = domain_rep + self.dropout(ffn_output)
        domain_rep = self.norm2(domain_rep)
        
        weights = self.weight_proj(domain_rep)
        weights = F.softmax(weights, dim=-1)
        
        return weights


class DAMMFND_TEXT_MODEL(torch.nn.Module):
    """纯文本版 DAMMFND"""
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout):
        super(DAMMFND_TEXT_MODEL, self).__init__()
        
        self.num_expert = 6
        self.task_num = 2
        self.domain_num = self.task_num
        self.gate_num = 3
        self.num_share = 1
        
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        self.text_dim = 768
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        
        # Domain-Specific Experts
        text_expert_list = []
        for i in range(self.domain_num):
            text_expert = []
            for j in range(self.num_expert):
                text_expert.append(cnn_extractor(emb_dim, feature_kernel))
            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
        self.text_experts = nn.ModuleList(text_expert_list)
        
        text_expert_list2 = []
        for i in range(self.domain_num):
            text_expert = []
            for j in range(self.num_expert):
                text_expert.append(cnn_extractor(emb_dim, feature_kernel))
            text_expert = nn.ModuleList(text_expert)
            text_expert_list2.append(text_expert)
        self.text_experts2 = nn.ModuleList(text_expert_list2)
        
        text_expert_list3 = []
        for i in range(self.domain_num):
            text_expert = []
            for j in range(self.num_expert):
                text_expert.append(cnn_extractor(emb_dim, feature_kernel))
            text_expert = nn.ModuleList(text_expert)
            text_expert_list3.append(text_expert)
        self.text_experts3 = nn.ModuleList(text_expert_list3)
        
        # Shared Experts
        text_share_expert = []
        for i in range(self.num_share):
            share = []
            for j in range(self.num_expert * 2):
                share.append(cnn_extractor(emb_dim, feature_kernel))
            share = nn.ModuleList(share)
            text_share_expert.append(share)
        self.text_share_expert = nn.ModuleList(text_share_expert)
        
        text_share_expert2 = []
        for i in range(self.num_share):
            share = []
            for j in range(self.num_expert * 2):
                share.append(cnn_extractor(emb_dim, feature_kernel))
            share = nn.ModuleList(share)
            text_share_expert2.append(share)
        self.text_share_expert2 = nn.ModuleList(text_share_expert2)
        
        text_share_expert3 = []
        for i in range(self.num_share):
            share = []
            for j in range(self.num_expert * 2):
                share.append(cnn_extractor(emb_dim, feature_kernel))
            share = nn.ModuleList(share)
            text_share_expert3.append(share)
        self.text_share_expert3 = nn.ModuleList(text_share_expert3)
        
        # Fusion Experts
        fusion_expert_list = []
        for i in range(self.domain_num):
            fusion_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(
                    nn.Linear(320, 320),
                    nn.SiLU(),
                    nn.Linear(320, 320),
                )
                fusion_expert.append(expert)
            fusion_expert = nn.ModuleList(fusion_expert)
            fusion_expert_list.append(fusion_expert)
        self.fusion_experts = nn.ModuleList(fusion_expert_list)
        
        fusion_share_expert = []
        for i in range(self.num_share):
            share = []
            for j in range(self.num_expert * 2):
                expert = nn.Sequential(
                    nn.Linear(320, 320),
                    nn.SiLU(),
                    nn.Linear(320, 320),
                )
                share.append(expert)
            share = nn.ModuleList(share)
            fusion_share_expert.append(share)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
        
        # Gate Networks
        text_gate_list = []
        for i in range(self.domain_num):
            text_gate = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, self.num_expert * 3),
                nn.Dropout(0.1),
                nn.Softmax(dim=1)
            )
            text_gate_list.append(text_gate)
        self.text_gate_list = nn.ModuleList(text_gate_list)
        
        text_gate_list2 = []
        for i in range(self.domain_num):
            text_gate = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, self.num_expert * 3),
                nn.Dropout(0.1),
                nn.Softmax(dim=1)
            )
            text_gate_list2.append(text_gate)
        self.text_gate_list2 = nn.ModuleList(text_gate_list2)
        
        text_gate_list3 = []
        for i in range(self.domain_num):
            text_gate = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, self.num_expert * 3),
                nn.Dropout(0.1),
                nn.Softmax(dim=1)
            )
            text_gate_list3.append(text_gate)
        self.text_gate_list3 = nn.ModuleList(text_gate_list3)
        
        fusion_gate_list0 = []
        for i in range(self.domain_num):
            fusion_gate = nn.Sequential(
                nn.Linear(320, 160),
                nn.SiLU(),
                nn.Linear(160, self.num_expert * 3),
                nn.Dropout(0.1),
                nn.Softmax(dim=1)
            )
            fusion_gate_list0.append(fusion_gate)
        self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
        
        # Attention Modules
        self.text_attention = MaskAttention(emb_dim)
        self.text_attention3 = MaskAttention(emb_dim)
        
        # Classifiers
        self.text_classifier = MLP(320, mlp_dims, dropout)
        self.text_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)
        self.text_classifier2 = MLP(320, mlp_dims, dropout)
        self.text_classifier_Mu2 = MLP_Mu(320, mlp_dims, dropout)
        self.text_classifier3 = MLP(320, mlp_dims, dropout)
        self.text_classifier_Mu3 = MLP_Mu(320, mlp_dims, dropout)
        
        self.domain_aware_text_classifier = MLP(320, mlp_dims, dropout)
        self.domain_aware_text_classifier2 = MLP(320, mlp_dims, dropout)
        self.domain_aware_text_classifier3 = MLP(320, mlp_dims, dropout)
        self.domain_aware_total_classifier = MLP(320, mlp_dims, dropout)
        self.domain_aware_total_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)
        
        self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(320, 320, [348], 0.1)
        self.att_mlp_text = MLP_fusion(320, 2, [174], 0.1)
        self.att_mlp_text2 = MLP_fusion(320, 2, [174], 0.1)
        self.att_mlp_text3 = MLP_fusion(320, 2, [174], 0.1)
        
        self.fake_news_layernorm = LayerNorm(320 * 3, eps=1e-12)
        self.domain_classification_layernorm = LayerNorm(320 * 1, eps=1e-12)
        
        self.gate_text_prefer = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )
        self.gate_text_prefer2 = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )
        self.gate_text_prefer3 = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )
        
        self.attention = DomainAwareTransformer(dim=320, num_heads=8)
        self.tau = 0.5

    def _get_three_views(self, text_feature, masks):
        view1 = text_feature[:, 0, :]
        mask_expanded = masks.unsqueeze(-1).float()
        view2 = (text_feature * mask_expanded).sum(dim=1) / masks.sum(dim=1, keepdim=True).float()
        view3 = self.text_attention3(text_feature, masks)
        return view1, view2, view3

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        
        text_feature = self.bert(inputs, attention_mask=masks)[0]
        view1, view2, view3 = self._get_three_views(text_feature, masks)
        
        # View 1
        text_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list[i](view1)
            text_gate_out_list.append(gate_out)
        
        text_gate_expert_value = []
        text_experts_feature = 0
        text_gate_share_expert_value = []
        for i in range(1):
            gate_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.text_experts[i][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert * 2):
                tmp_expert = self.text_share_expert[0][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
            text_experts_feature = gate_expert
            text_gate_share_expert_value.append(gate_share_expert)
        
        att = F.softmax(self.att_mlp_text(text_experts_feature), dim=-1)
        text_experts_feature0 = att[:, 0].view(-1, 1) * text_experts_feature
        text_experts_feature1 = att[:, 1].view(-1, 1) * text_experts_feature
        text_gate_expert_value.append(text_experts_feature0)
        text_gate_expert_value.append(text_experts_feature1)
        
        # View 2
        text_gate_out_list2 = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list2[i](view2)
            text_gate_out_list2.append(gate_out)
        
        text_gate_expert_value2 = []
        text_experts_feature2 = 0
        for i in range(1):
            gate_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.text_experts2[i][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list2[i][:, j].unsqueeze(1))
            for j in range(self.num_expert * 2):
                tmp_expert = self.text_share_expert2[0][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list2[i][:, (self.num_expert + j)].unsqueeze(1))
            text_experts_feature2 = gate_expert
        
        att2 = F.softmax(self.att_mlp_text2(text_experts_feature2), dim=-1)
        text_experts_feature2_0 = att2[:, 0].view(-1, 1) * text_experts_feature2
        text_experts_feature2_1 = att2[:, 1].view(-1, 1) * text_experts_feature2
        text_gate_expert_value2.append(text_experts_feature2_0)
        text_gate_expert_value2.append(text_experts_feature2_1)
        
        # View 3
        text_gate_out_list3 = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list3[i](view3)
            text_gate_out_list3.append(gate_out)
        
        text_gate_expert_value3 = []
        text_experts_feature3 = 0
        for i in range(1):
            gate_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.text_experts3[i][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list3[i][:, j].unsqueeze(1))
            for j in range(self.num_expert * 2):
                tmp_expert = self.text_share_expert3[0][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list3[i][:, (self.num_expert + j)].unsqueeze(1))
            text_experts_feature3 = gate_expert
        
        att3 = F.softmax(self.att_mlp_text3(text_experts_feature3), dim=-1)
        text_experts_feature3_0 = att3[:, 0].view(-1, 1) * text_experts_feature3
        text_experts_feature3_1 = att3[:, 1].view(-1, 1) * text_experts_feature3
        text_gate_expert_value3.append(text_experts_feature3_0)
        text_gate_expert_value3.append(text_experts_feature3_1)
        
        # Fusion
        fusion_share_feature = torch.cat((
            text_gate_share_expert_value[0],
            text_gate_expert_value2[0],
            text_gate_expert_value3[0]
        ), dim=-1)
        fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        fusion_gate_input0 = self.domain_fusion(fusion_share_feature)
        
        fusion_gate_out_list0 = []
        for k in range(self.domain_num):
            gate_out = self.fusion_gate_list0[k](fusion_gate_input0)
            fusion_gate_out_list0.append(gate_out)
        
        fusion_gate_expert_value0 = []
        fusion_experts_feature = 0
        fusion_gate_share_expert_value0 = []
        for m in range(1):
            share_gate_expert0 = 0
            gate_share_expert = 0
            for n in range(self.num_expert):
                fusion_tmp_expert0 = self.fusion_experts[m][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * fusion_gate_out_list0[m][:, n].unsqueeze(1))
            for n in range(self.num_expert * 2):
                fusion_tmp_expert0 = self.fusion_share_expert[0][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
                gate_share_expert += (fusion_tmp_expert0 * fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
            fusion_gate_share_expert_value0.append(gate_share_expert)
            fusion_experts_feature = fusion_tmp_expert0
        
        att_fusion = F.softmax(self.att_mlp_text(fusion_experts_feature), dim=-1)
        fusion_experts_feature0 = att_fusion[:, 0].view(-1, 1) * fusion_experts_feature
        fusion_experts_feature1 = att_fusion[:, 1].view(-1, 1) * fusion_experts_feature
        fusion_gate_expert_value0.append(fusion_experts_feature0)
        fusion_gate_expert_value0.append(fusion_experts_feature1)
        
        # Classifiers
        text_two_task = []
        text_two_task.append(self.text_classifier(text_gate_expert_value[0]).squeeze(1))
        text_two_task.append(self.text_classifier_Mu(text_gate_expert_value[1]).squeeze(1))
        
        text_two_task2 = []
        text_two_task2.append(self.text_classifier2(text_gate_expert_value2[0]).squeeze(1))
        text_two_task2.append(self.text_classifier_Mu2(text_gate_expert_value2[1]).squeeze(1))
        
        text_two_task3 = []
        text_two_task3.append(self.text_classifier3(text_gate_expert_value3[0]).squeeze(1))
        text_two_task3.append(self.text_classifier_Mu3(text_gate_expert_value3[1]).squeeze(1))
        
        fusion_two_task = []
        fusion_two_task.append(self.domain_aware_total_classifier(fusion_gate_expert_value0[0]).squeeze(1))
        fusion_two_task.append(self.domain_aware_total_classifier_Mu(fusion_gate_expert_value0[1]).squeeze(1))
        
        # Disentanglement
        text_fake_news = torch.softmax(text_two_task[0], -1)
        text_multi_domain = torch.softmax(text_two_task[1], -1)
        text_fake_news2 = torch.softmax(text_two_task2[0], -1)
        text_multi_domain2 = torch.softmax(text_two_task2[1], -1)
        text_fake_news3 = torch.softmax(text_two_task3[0], -1)
        text_multi_domain3 = torch.softmax(text_two_task3[1], -1)
        fusion_fake_news = torch.softmax(fusion_two_task[0], -1)
        fusion_multi_domain = torch.softmax(fusion_two_task[1], -1)
        
        # Discriminator
        multi_label_feature = (
            text_gate_expert_value[0] + 
            text_gate_expert_value2[0] + 
            text_gate_expert_value3[0]
        )
        fake_news_feature = (
            text_gate_expert_value[1] + 
            text_gate_expert_value2[1] + 
            text_gate_expert_value3[1]
        )
        
        text_domain_features = text_gate_expert_value[0]
        text_domain_features = self.gate_text_prefer(fake_news_feature) * text_domain_features
        
        text_domain_features2 = text_gate_expert_value2[0]
        text_domain_features2 = self.gate_text_prefer2(fake_news_feature) * text_domain_features2
        
        text_domain_features3 = text_gate_expert_value3[0]
        text_domain_features3 = self.gate_text_prefer3(fake_news_feature) * text_domain_features3
        
        domain_aware_text_view = torch.sigmoid(
            self.domain_aware_text_classifier(text_gate_expert_value[0] + text_domain_features).squeeze()
        )
        domain_aware_text_view2 = torch.sigmoid(
            self.domain_aware_text_classifier2(text_gate_expert_value2[0] + text_domain_features2).squeeze()
        )
        domain_aware_text_view3 = torch.sigmoid(
            self.domain_aware_text_classifier3(text_gate_expert_value3[0] + text_domain_features3).squeeze()
        )
        
        # Decision Layer
        weight_common = self.attention(
            [text_gate_expert_value[0], text_gate_expert_value2[0], text_gate_expert_value3[0]], 
            multi_label_feature
        )
        
        fake_news_sigmoid = (
            weight_common[:, 0].squeeze() * domain_aware_text_view +
            weight_common[:, 1].squeeze() * domain_aware_text_view2 +
            weight_common[:, 2].squeeze() * domain_aware_text_view3
        )
        
        fake_news_sigmoid = torch.clamp(fake_news_sigmoid, min=0.0, max=1.0)
        
        return (
            fake_news_sigmoid,
            text_fake_news, text_multi_domain,
            text_fake_news2, text_multi_domain2,
            text_fake_news3, text_multi_domain3,
            fusion_fake_news, fusion_multi_domain,
            domain_aware_text_view, domain_aware_text_view2, domain_aware_text_view3
        )


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 loss_weight=[1, 0.006, 0.009, 5e-5],
                 early_stop=5,
                 epoches=100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        
        if not os.path.exists(save_param_dir):
            os.makedirs(save_param_dir)
        self.save_param_dir = save_param_dir

    def train(self):
        self.model = DAMMFND_TEXT_MODEL(self.emb_dim, self.mlp_dims, self.bert, 320, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        recorder = Recorder(self.early_stop)
        
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                category = batch_data['multi_category']
                labels_domain = category
                
                outputs = self.model(**batch_data)
                (fake_news_sigmoid,
                 text_fake_news, text_multi_domain,
                 text_fake_news2, text_multi_domain2,
                 text_fake_news3, text_multi_domain3,
                 fusion_fake_news, fusion_multi_domain,
                 domain_aware_text_view, domain_aware_text_view2, domain_aware_text_view3) = outputs
                
                loss0 = loss_fn(fake_news_sigmoid, label.float())
                
                loss11 = F.binary_cross_entropy_with_logits(text_multi_domain, labels_domain.float())
                loss21 = F.binary_cross_entropy_with_logits(text_multi_domain2, labels_domain.float())
                loss31 = F.binary_cross_entropy_with_logits(text_multi_domain3, labels_domain.float())
                loss41 = F.binary_cross_entropy_with_logits(fusion_multi_domain, labels_domain.float())
                
                loss12_aux = loss_fn(domain_aware_text_view.squeeze(), label.float())
                loss22_aux = loss_fn(domain_aware_text_view2.squeeze(), label.float())
                loss32_aux = loss_fn(domain_aware_text_view3.squeeze(), label.float())
                
                uniform_target = torch.ones_like(text_fake_news, dtype=torch.float, device=text_fake_news.device) / 9
                loss12 = F.kl_div(text_fake_news, uniform_target.float())
                loss22 = F.kl_div(text_fake_news2, uniform_target.float())
                loss32 = F.kl_div(text_fake_news3, uniform_target.float())
                loss42 = F.kl_div(fusion_fake_news, uniform_target.float())
                
                loss = (loss0 + 
                       (loss11 + loss12 + loss21 + loss22 + loss31 + loss32 + loss41 + loss42) / 8 + 
                       (loss12_aux + loss22_aux + loss32_aux) / 3.0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                
                avg_loss.add(loss.item())
            
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results0 = self.test(self.val_loader)
            print('Training Loss: {}'.format(avg_loss.item()))
            print('Final AUC: {}'.format(results0.get('auc', 'N/A')))
            print('Final Macro F1 / metric: {}'.format(results0.get('metric', 'N/A')))
            print('Final Recall: {}'.format(results0.get('recall', 'N/A')))
            print('Final Precision: {}'.format(results0.get('precision', 'N/A')))
            print('Final Accuracy: {}'.format(results0.get('acc', 'N/A')))
            mark = recorder.add(results0)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                          os.path.join(self.save_param_dir, 'parameter_dammfnd_text.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_dammfnd_text.pkl')))
        print("开始进行最后的测试")
        results0 = self.test(self.test_loader)
        print("final: ", results0)
        
        return results0, os.path.join(self.save_param_dir, 'parameter_dammfnd_text.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                
                outputs = self.model(**batch_data)
                batch_label_pred = outputs[0]
                
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        metric_res = metricsTrueFalse(label, pred, category, self.category_dict)
        return metric_res