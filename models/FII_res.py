import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.CBiAFormer import DLETransformer, DMS_base_k3_window12_384, DMS_tiny_k3_window7_224
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import CrossEntropyLoss
from ASLmain.src.loss_functions.losses import AsymmetricLoss
from torchvision.transforms import InterpolationMode
from apex import amp

class TopKMaxPooling(nn.Module):
    def __init__(self, kmax=1.0):
        super(TopKMaxPooling, self).__init__()
        self.kmax = kmax

    @staticmethod
    def get_positive_k(k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, input):
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)
        n = h * w  # number of regions
        kmax = self.get_positive_k(self.kmax, n)
        sorted, indices = torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True)
        region_max = sorted.narrow(2, 0, kmax)
        output = region_max.sum(2).div_(kmax)
        return output.view(batch_size, num_channels)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ')'

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.in_channels = in_features
        self.out_channels = in_features or out_features
        self.hidden_channels = hidden_features
        self.fc1 = nn.Linear(self.in_channels, self.hidden_channels)
        self.bn1 = nn.LayerNorm(self.hidden_channels)
        self.act = act_layer()

        self.fc2 = nn.Linear(self.hidden_channels, self.out_channels)
        self.bn2 = nn.LayerNorm(self.out_channels)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.drop2 = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.drop2(x)

        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.weight = nn.Conv1d(in_channels, out_channels, 1)
        self.act = nn.GELU()

    def forward(self, adj, inputs):
        # adj: [N*N]  inputs: [C*N]  output: [B, C, N]
        x = torch.matmul(inputs, adj)
        x = self.act(x)
        x = self.weight(x)
        x = self.act(x)
        return x


class Transformer(nn.Module):
    def __init__(self, feature_dim=384, n_head=8, layer_num=6, dropout=0.1, drop_path = 0.1,
                 mlp_ratio=4, act=nn.GELU):
        super(Transformer, self).__init__()

        self.feature_dim = feature_dim
        self.n_head = n_head
        self.layer_num = layer_num
        layers = []
        # build blocks
        for i in range(layer_num):
            layers += [EncoderLayer(
                feature_dim=self.feature_dim,
                n_head=self.n_head,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                act=act,
                drop_path = drop_path)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, feature_dim, n_head, dropout=0.1, mlp_ratio=4, act=nn.GELU, drop_path=0.1):
        super(EncoderLayer, self).__init__()
        self.multiheadattention = nn.MultiheadAttention(feature_dim, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(feature_dim)
        #self.dropout = nn.Dropout(dropout)
        self.mlp = Mlp(feature_dim, feature_dim * mlp_ratio, drop=dropout, act_layer=nn.GELU)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        '''
        :param x: [B, N, C]
        :return: [B, N, C]
        '''
        shortcut = x  # B N C
        x = self.norm1(x)

        x = x.permute(1, 0, 2)  # N，B，C
        x, _ = self.multiheadattention(x, x, x)
        x = x.permute(1, 0, 2)  # B, N, C
        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, hidden_dim, head=4, dropout=0.1, drop_path = 0.1, mlp_ratio=4.0):
        super(Cross_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.MultiheadAttention(self.hidden_dim, head, dropout=dropout)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.drop_path = DropPath(drop_path)
        self.mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(self.hidden_dim, self.mlp_hidden_dim)

    def forward(self, src1, src2):
        B, C, H, W = src1.shape
        src1 = src1.view(B, C, -1).permute(0, 2, 1)  # B, L, C
        shortcut = src1
        src1 = self.norm(src1)
        src1 = src1.permute(1, 0,
                            2)  # (L, N, E)` where L is the target sequence length, N is the batch size, E is the embedding dimension.
        src2 = src2.view(B, C, -1)
        src2 = src2.permute(2, 0,
                            1)  # (S, N, E)`, where S is the source sequence length, N is the batch size, E is the embedding dimension.
        out, attn = self.attn(query=src1,
                              key=src2,
                              value=src2)  # (L, N, E)` where L is the target sequence length, N is the batch size,
        out = out.permute(1, 0, 2)
        x = self.drop_path(out) + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x) + shortcut
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x


class Cross_Attention_Layer(nn.Module):
    def __init__(self, hidden_dim, feature_dim, use_cross_attention, drop_path=0.1):
        super(Cross_Attention_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.use_cross_attn = use_cross_attention

        self.conv28x28 = BasicConv(feature_dim // 4, hidden_dim, 1, 1, 0)
        self.conv14x14 = BasicConv(feature_dim // 2, hidden_dim, 1, 1, 0)
        self.conv7x7 = BasicConv(feature_dim, hidden_dim, 1, 1, 0)

        if self.use_cross_attn:
            self.cross28x14 = Cross_Attention(hidden_dim, drop_path = drop_path)
            self.cross28x7 = Cross_Attention(hidden_dim, drop_path = drop_path)
            self.cross14x28 = Cross_Attention(hidden_dim, drop_path = drop_path)
            self.cross14x7 = Cross_Attention(hidden_dim, drop_path = drop_path)
            self.cross7x28 = Cross_Attention(hidden_dim, drop_path = drop_path)
            self.cross7x14 = Cross_Attention(hidden_dim, drop_path = drop_path)

            self.Rconv28 = BasicConv(3 * hidden_dim, hidden_dim, 1, 1, 0)
            self.Rconv14 = BasicConv(3 * hidden_dim, hidden_dim, 1, 1, 0)
            self.Rconv7 = BasicConv(3 * hidden_dim, hidden_dim, 1, 1, 0)

    def forward(self, f1, f2, f3):
        c1 = self.conv28x28(f1)
        c2 = self.conv14x14(f2)
        c3 = self.conv7x7(f3)

        if not self.use_cross_attn:
            return c1, c2, c3

        f12 = self.cross28x14(c1, c2)
        f13 = self.cross28x7(c1, c3)
        # print(f12.shape)

        f21 = self.cross14x28(c2, c1)
        f23 = self.cross14x7(c2, c3)
        # print(f21.shape)

        f31 = self.cross7x28(c3, c1)
        f32 = self.cross7x14(c3, c2)
        # print(f31.shape)

        r1 = torch.cat((c1, f12, f13), dim=1)
        r1 = self.Rconv28(r1)
        r2 = torch.cat((c2, f21, f23), dim=1)
        r2 = self.Rconv14(r2)
        r3 = torch.cat((c3, f31, f32), dim=1)
        r3 = self.Rconv7(r3)

        return r1, r2, r3


class FeaturePyramid(nn.Module):
    def __init__(self, feature_dim, multi_numclass):
        super(FeaturePyramid, self).__init__()
        self.feature_dim = feature_dim
        self.smooth1 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)

        self.kmp = TopKMaxPooling(0.05)
        self.fc1 = nn.Conv2d(feature_dim, multi_numclass, 1, 1, 0)
        self.fc2 = nn.Conv2d(feature_dim, multi_numclass, 1, 1, 0)
        self.fc3 = nn.Conv2d(feature_dim, multi_numclass, 1, 1, 0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, f1, f2, f3):
        p3 = f3
        p2 = self._upsample_add(p3, f2)
        p1 = self._upsample_add(p2, f1)

        score3 = self.kmp(self.fc3(self.smooth3(p3)))
        score2 = self.kmp(self.fc2(self.smooth2(p2)))
        score1 = self.kmp(self.fc1(self.smooth1(p1)))

        return score1, score2, score3, p1


class FICI(nn.Module):
    def __init__(self, model, model_type, num_classes=172, num_multi_classes=353,
                 n_head=4, layer_num=3, adj_file="E:/Code/MM/adj_172.pkl", drop_rate=0.1, drop_path = 0.1,
                 use_cross_attention=False, use_adj=False, use_gcn=False, use_cross_modelity_attention= False):
        super(FICI, self).__init__()
        self.model = model
        self.model_type = model_type
        self.num_classes = num_classes
        self.num_multi_classes = num_multi_classes
        self.adj_file = adj_file

        self.use_adj = use_adj
        self.use_gcn = use_gcn
        self.use_cross_attention = use_cross_attention
        self.use_cross_modelity_attention = use_cross_modelity_attention

        if "resnet" in model_type:
            self.layer1 = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
            )
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.pos = None
            self.embed = None
            self.feature_dim = 2048
            self.backbone = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4, self.pos, self.embed])
            self.hidden_dim = self.feature_dim // 2

        elif "cart" in model_type:
            self.pos = model.absolute_pos_embed
            self.embed = model.patch_embed
            self.layers = model.layers
            self.pos_drop = model.pos_drop
            self.feature_dim = 1024 if "base" in model_type else 768
            self.norm = model.norm
            self.backbone = nn.ModuleList([self.pos, self.embed, self.layers, self.pos_drop, self.norm])
            self.hidden_dim = self.feature_dim // 2

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.kmp = TopKMaxPooling(0.05)

        self.cross_attention = Cross_Attention_Layer(hidden_dim=self.hidden_dim, feature_dim=self.feature_dim,
                                                     use_cross_attention=self.use_cross_attention, drop_path=drop_path)
        self.fpn = FeaturePyramid(feature_dim=self.hidden_dim, multi_numclass=self.num_multi_classes)


        self.class_fc = nn.Linear(self.feature_dim, num_classes)
        self.class_conv = nn.Linear(self.feature_dim, self.hidden_dim)

        #self.ing_conv = nn.Conv2d(self.feature_dim, self.hidden_dim, 3, 1, 1)
        self.ingredient_fc = nn.Conv2d(self.hidden_dim, self.num_multi_classes, 1, 1, 0)

        self.ingredient_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, 1, 0)
        self.activation = nn.Conv2d(self.hidden_dim, self.num_multi_classes, 1, 1, 0)

        if self.use_gcn:
            self.transformer = Transformer(feature_dim=self.hidden_dim, n_head=n_head, layer_num=layer_num,
                                           dropout=drop_rate, drop_path = drop_path)

            #adj = self.get_adj(0.5, adj_file)
            #self.adj_A = nn.Parameter(torch.from_numpy(adj).float())
            if self.use_adj:
                adj = self.get_adj(0.5, adj_file)
                self.adj_A = nn.Parameter(torch.from_numpy(adj).float())
                self.gcn1 = GCN(in_channels=self.hidden_dim, out_channels=self.hidden_dim)


            self.global_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, 1)
            self.global_bn = nn.BatchNorm1d(self.hidden_dim)
            self.relu1 = nn.LeakyReLU(0.2)

            self.global_cate_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, 1)
            self.global_cate_bn = nn.BatchNorm1d(self.hidden_dim)
            self.relu2 = nn.LeakyReLU(0.2)

            self.transform_conv = nn.Conv1d(self.hidden_dim * 2, self.num_multi_classes, 1)

            #self.mconv1 = nn.Conv2d(self.feature_dim, self.num_multi_classes, 3, 1, 1)
            #self.mconv2 = nn.Conv2d(self.hidden_dim, self.num_multi_classes, 4, 4, 0)
            self.shortcut = nn.Conv1d(self.hidden_dim, self.feature_dim, 1)
            self.gcn2 = GCN(in_channels=self.hidden_dim, out_channels=self.feature_dim)

            self.mask_mat = nn.Parameter(torch.eye(self.num_multi_classes).float())
            self.gcn_fc = nn.Conv1d(self.feature_dim, self.num_multi_classes, 1)

        else:
            self.gcn_fc = nn.Linear(self.hidden_dim, self.num_multi_classes)

        if self.use_cross_modelity_attention:
            self.icattn = nn.MultiheadAttention(self.feature_dim, 8, dropout=drop_rate)
            self.norm1 = nn.LayerNorm(self.feature_dim)
            self.fc2 = nn.Linear(self.feature_dim, num_classes)

        amp.register_float_function(torch, 'sigmoid') 
        amp.register_float_function(torch, 'softmax')


    def gen_ingredient_node(self, x, mask):
        #print(x.shape)
        #mask = self.activation(x)
        #loss = self.kmp(mask)
        #mask = x
        mask = mask.view(mask.size(0), mask.size(1), -1)  # (B, C_num, H*W)
        mask = torch.sigmoid(mask)  # (B, C_num, H*W) - sigmoid
        mask = mask.transpose(1, 2)  # (B, HW, Num_class)

        x = self.ingredient_conv(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def gen_adjust_graph(self, node, ing_feature, category_feature):
        
        ing_guid = self.global_conv(ing_feature)
        ing_guid = self.global_bn(ing_guid)
        ing_guid = self.relu1(ing_guid)
        ing_guid = ing_guid.expand(ing_guid.size(0), ing_guid.size(1), node.size(2))

        #category_features = category_feature.detach()
        #cate_guid = self.global_cate_conv(category_features)
        #cate_guid = self.global_cate_bn(cate_guid)
        #cate_guid = self.relu2(cate_guid)
        #cate_guid = cate_guid.expand(cate_guid.size(0), cate_guid.size(1), node.size(2))

        node = torch.cat((ing_guid, node), dim=1)
        #node = torch.cat((ing_guid, cate_guid, node), dim=1)
        graph = self.transform_conv(node)
        graph = torch.sigmoid(graph)

        return graph

    '''
    def gen_adjust_graph(self, category_f, ingredient_f):
        # print(ingredient_f.shape)
        category_f = self.mconv1(category_f)
        category_f = category_f.view(category_f.size(0), category_f.size(1), -1)
        # print(category_f.shape)

        ingredient_f = self.mconv2(ingredient_f)
        # print(ingredient_f.shape)
        ingredient_f = ingredient_f.view(ingredient_f.size(0), ingredient_f.size(1), -1)
        # print(ingredient_f.shape)
        # print(category_f.shape)
        # print(ingredient_f.shape)
        graph = category_f @ ingredient_f.permute(0, 2, 1)
        graph = torch.sigmoid(graph)
        # print(graph.shape)

        return graph
    '''
    def get_adj(self, k, adj_file):
        '''
        :param k: the threshold. default 0.4.
        :param adj_file: the file path of adj_matrix
        :return:
        '''
        import pickle
        import numpy as np
        adj_mat = pickle.load(open(adj_file, 'rb'))
        adj = adj_mat['adj']
        nums = adj_mat['nums']
        nums = nums[:, np.newaxis]
        adj = adj / nums
        adj[adj < k] = 0
        adj[adj >= k] = 1
        adj = adj * 0.25 / (adj.sum(0, keepdims=True) + 1e-6)
        adj = adj + np.identity(self.num_multi_classes, np.int)

        return adj

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def forward_features(self, x):
        if "resnet" in self.model_type:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return x2, x3, x4

        elif "cart" in self.model_type:
            x = self.embed(x)
            pos = self.pos(x)
            x = pos + x
            x = self.pos_drop(x)
            out = []
            # print(x.shape)
            for name, module in self.layers.named_children():
                pre = None
                for sub_name, sub_module in module.named_children():
                    # print(sub_name)
                    x = sub_module(x)
                    if "downsample" in sub_name:
                        out.append(pre)
                    pre = x
                if name == "3":
                    out.append(pre)

            return out[1], out[2], out[3]

    def forward_score(self, x):
        # return the feature vector and the category predicted score
        if "resnet" in self.model_type:
            x = self.avg(x)
            feature = torch.flatten(x, 1)
            x = self.class_fc(feature)
            return feature, x
        elif "cart" in self.model_type:
            x = self.norm(x)  # B C H W
            x = self.avg(x)  # B C 1
            feature = torch.flatten(x, 1)
            #print(feature.shape)
            x = self.class_fc(feature)
            return feature, x

    def forward(self, x, label=None, multi_label=None):
        B = x.size(0)
        # computing the features of different stages and the category score.
        f3, f4, f5 = self.forward_features(x)
        # computing the category-level feature vector and predicted score.
        cf, cs = self.forward_score(f5)
        #print(cs)
        # computing the ingredient vector (B, 1, C1)
        r1, r2, r3 = self.cross_attention(f3, f4, f5)
        s1, s2, s3, f = self.fpn(r1, r2, r3)
        mask = self.ingredient_fc(f)
        ingredient_logits = self.kmp(mask)

        if self.use_gcn:
            # compute the category features (B, C) -> (B, 1, C1)
            category_feature = self.class_conv(cf)
            category_feature = torch.unsqueeze(category_feature, dim=2)
            # print(category_feature.shape)

            ingredient_feature = self.kmp(f)
            ingredient_feature = torch.unsqueeze(ingredient_feature, dim=2)

            #ing_f = self.ing_conv(f5)

            node = self.gen_ingredient_node(f, mask)
            

            final_node = torch.cat((category_feature, ingredient_feature, node), dim=2)
            final_node = final_node.permute((0,2,1))
            # (B, Num_multi_class+2, C1)
            # print(node.shape)

            # computing the node of ingredient features
            node = self.transformer(final_node)
            node = node[:, 2:, :]

            # Static GCN: use the co-occurrence relationship.
            if self.use_adj:
                node = node.permute(0, 2, 1)
                # print(node.shape)
                static_graph = self.gen_adj(self.adj_A).detach()
                out1 = self.gcn1(static_graph, node) + node
            else:
                out1 = node.permute(0, 2, 1)
                #out1 = node

            shortcut = self.shortcut(out1)

            #dynamic_graph = self.gen_adjust_graph(f5, f)
            #dynamic_graph = self.gen_adjust_graph(out1, ingredient_feature, category_feature)
            #out2 = self.gcn2(dynamic_graph, out1) + shortcut
            out2 = shortcut

            gcn_out = self.gcn_fc(out2)
            mask_mat = self.mask_mat.detach()
            gcn_out = (gcn_out * mask_mat).sum(-1)

            if self.use_cross_modelity_attention:
                final_cf, _ = self.icattn(query=cf.unsqueeze(dim=1).permute(1, 0, 2),
                                          key=out2.permute(2, 0, 1),
                                          value=out2.permute(2, 0, 1))
                final_cf = final_cf.permute(1, 0, 2).squeeze(dim=1)
                final_cf = self.norm1(final_cf) + cf
                final_cf = self.fc2(final_cf)

            else:
                final_cf = cs

        else:
            gcn_out = ingredient_logits
            final_cf = cs

        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss_multi = AsymmetricLoss(gamma_neg=4, gamma_pos=2, clip=0.05)
            #loss_bce = nn.BCEWithLogitsLoss(reduction="mean")
            loss_category = loss_fct(cs.view(-1, self.num_classes), label.view(-1))

            if self.use_cross_modelity_attention:
                loss_category_final = loss_fct(final_cf.view(-1, self.num_classes), label.view(-1))
                loss_fc = loss_category + loss_category_final
            else:
                loss_fc = loss_category

            #print(loss_fc)
            # print(loss_total)

            if multi_label is not None:
                loss_ing = loss_multi(gcn_out.view(-1, self.num_multi_classes), multi_label.float())
                if self.use_gcn:
                    loss_inc = loss_multi(ingredient_logits.view(-1, self.num_multi_classes), multi_label.float())
                    #loss_activation = loss_bce(activation_logits.view(-1, self.num_multi_classes), multi_label.float())
                else:
                    loss_inc = 0
                    #loss_activation = 0

                #print(loss_inc)
                loss_s1 = loss_multi(s1.view(-1, self.num_multi_classes), multi_label.float())
                loss_s2 = loss_multi(s2.view(-1, self.num_multi_classes), multi_label.float())
                loss_s3 = loss_multi(s3.view(-1, self.num_multi_classes), multi_label.float())
                loss_multis = loss_ing + loss_inc + loss_s1 + loss_s2 + loss_s3
                #loss_multis = loss_ing + loss_inc
                # loss_total = 0.5 * (
                # torch.exp(-self.w1) * loss_fc + torch.exp(-self.w2) * loss_multi + self.w1 + self.w2)
                loss_total = loss_fc +  0.2 * loss_multis

            return loss_total, final_cf, gcn_out + ingredient_logits, loss_fc, loss_multis

        else:
            final_predict = cs + final_cf
            return final_cf, cs, final_predict, gcn_out + ingredient_logits

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.backbone.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.backbone.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]


def build_models(model_type, num_class, multi_num_class, pretrained=None, pretrained_dir=None, adj_file=None,
                 use_cross_attention=True, use_adj=False, use_gcn=True, use_cross_modelity_attention =True, drop_path = 0.1, **kwargs):
    if "cart-base" in model_type:
        net = DMS_base_k3_window12_384(pretrained, pretrained_dir, **kwargs)
        model = FICI(net, model_type, num_classes=num_class, num_multi_classes=multi_num_class,adj_file=adj_file,
                     use_cross_attention=use_cross_attention, use_adj=use_adj, use_gcn=use_gcn,
                     use_cross_modelity_attention = use_cross_modelity_attention, drop_path = drop_path)
    else:
        net = DMS_tiny_k3_window7_224(pretrained, pretrained_dir, **kwargs)
        model = FICI(net, model_type, num_classes=num_class, num_multi_classes=multi_num_class,adj_file=adj_file,
                     use_cross_attention=use_cross_attention, use_adj=use_adj, use_gcn=use_gcn,
                     use_cross_modelity_attention = use_cross_modelity_attention, drop_path = drop_path)

    return model


'''
torch.set_printoptions(profile="full")
model = build_models("cart", 172, 353, True, "E:/Pretrained_model/best_82.4.pth", "E:/Code/MM/adj_172.pkl")
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
inputs = torch.randn([3,3,224,224])
label = torch.ones([3]).long()
multi_label = torch.ones([3,353]).long()
print(model(inputs, label, multi_label)[2].shape)
'''