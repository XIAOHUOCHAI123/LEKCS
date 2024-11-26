import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import dgl
from dgl.nn.pytorch import GraphConv
from transformer.layer_norm import LayerNorm
from transformer.multi_head_attention import MultiHeadAttention
from transformer.position_wise_feed_forward import PositionwiseFeedForward
from einops import repeat
from transformers import (
    LlamaTokenizer
)
from llama.modeling_llama import LlamaForSequenceClassification
from llama.mi_estimators import *
from llama.get_sentence_simi import SimiCal
from transformers import StoppingCriteria, StoppingCriteriaList
import warnings 
warnings.filterwarnings("ignore")

configs= {
    "num_query_token":8,
    "layer_num":1,
    "topk_c":18,
    "topk_f":18*3
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = get_device()

model_id = "path to llama"
tokenizer = LlamaTokenizer.from_pretrained(model_id)
llmmodel = LlamaForSequenceClassification.from_pretrained(model_id,num_labels=8, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
for param in llmmodel.parameters():
    param.requires_grad = False


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
#         self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation,allow_zero_in_degree=True))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation,allow_zero_in_degree=True))
        # output layer
#         self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, g,h):
#         h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class EncoderLayer(nn.Module):#transformer layer
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        #self.dropout2 = nn.Dropout(p=drop_prob)
    def forward(self, x, src_mask=None):
        # 1. compute self attention
        _x = x
        x,attention = self.attention(q=x, k=x, v=x, mask=src_mask)
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 4. add and norm
        #x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x,attention

class Model_GL(nn.Module):
    def __init__(self):
        super(Model_GL, self).__init__()
        #body,face,place,text,video,audio
        self.body_fc1  = nn.Linear(2048,1024)
        self.face_fc1  = nn.Linear(512,1024)
        self.text_fc1  = nn.Linear(768 ,1024)
        self.place_fc1 = nn.Linear(2048,1024)
        self.audio_fc1 = nn.Linear(1024,1024)
        self.video_fc1 = nn.Linear(2304,1024)
        
        self.gcn = GCN(in_feats=1024,n_hidden=1024,n_classes=8,n_layers=2,activation=F.relu,dropout=0.5)
        
        self.body_fc2  = nn.Linear(1024,512)
        self.face_fc2  = nn.Linear(1024,512)
        self.text_fc2  = nn.Linear(1024,512)
        self.place_fc2 = nn.Linear(1024,512)
        self.video_fc2 = nn.Linear(1024,512)
        self.audio_fc2 = nn.Linear(1024,512)
        
        self.body_lstm = nn.LSTM(512, 512, num_layers=1)
        self.face_lstm = nn.LSTM(512, 512, num_layers=1)
        self.text_lstm = nn.LSTM(512, 512, num_layers=1)
        self.place_lstm = nn.LSTM(512, 512, num_layers=1)

        
        self.relu      = nn.ReLU()
        self.insfc     = nn.Linear(512*8,1024)
        self.cls_token_c = nn.Parameter(torch.randn(1, 1, 1024))
        self.cls_token_f = nn.Parameter(torch.randn(1, 1, 512))
        #d_model, ffn_hidden, n_head, drop_prob
        self.coarse_attention_1 = EncoderLayer(d_model=1024, ffn_hidden=1024, n_head=2, drop_prob=0.2)
        self.coarse_attention_2 = EncoderLayer(d_model=1024, ffn_hidden=1024, n_head=2, drop_prob=0.2)
        self.coarse_attention_class = EncoderLayer(d_model=1024, ffn_hidden=1024, n_head=2, drop_prob=0.2)
        
        
        self.fine_attention_1 = EncoderLayer(d_model=512, ffn_hidden=512, n_head=2, drop_prob=0.2)
        self.fine_attention_2 = EncoderLayer(d_model=512, ffn_hidden=512, n_head=2, drop_prob=0.2)
        self.fine_attention_class = EncoderLayer(d_model=512, ffn_hidden=512, n_head=2, drop_prob=0.2)
        self.classfier_coarse = nn.Linear(1024,8)
        self.classfier_fine   = nn.Linear(512,8)

    def data2graph(self,places,body_feature,face_feature,text_feature,graphs,pframes,gids,batchsize):
        bid = 0
        fid = 0
        tid = 0
        for idx in range(batchsize):
            for i, graph in enumerate(graphs[idx]):
                features = []
                features.append(places[idx][i])
                frame = pframes[idx][i]
                gid   = gids[idx][i]
                for node in gid:
                    if node == "place":
                        continue
                    elif "_body" in node:
                        features.append(body_feature[bid])
                        bid = bid + 1
                    elif "_face" in node:
                        features.append(face_feature[fid])
                        fid = fid + 1
                    elif node == "text":
                        features.append(text_feature[tid])
                        tid = tid + 1
                features = torch.stack(features,dim=0)
                graph.ndata["feature"] = features
        return graphs
    def graph2data(self,places,bodys,faces,texts,graphs,pframes,batchsize):
        place_feature,body_feature,face_feature,text_feature = [],[],[],[]
        for idx in range(batchsize):
            place_s = []
            for i, graph in enumerate(graphs[idx]):
                data = graph.ndata["feature"]
                f_idx = 0
                place_s.append(data[f_idx])
                frame = pframes[idx][i]
                bps   = list(bodys[idx][frame].keys())
                for p in range(len(bps)):
                    f_idx = f_idx + 1
                    body_feature.append(data[f_idx])
                fps   = list(faces[idx][frame].keys())
                for p in range(len(fps)):
                    f_idx = f_idx + 1
                    face_feature.append(data[f_idx])
                if frame in texts[idx].keys():
                    f_idx = f_idx + 1
                    text_feature.append(data[f_idx])
            place_s = torch.stack(place_s,dim=0)
            place_feature.append(place_s)
        place_feature = torch.stack(place_feature,dim=0)
        body_feature  = torch.stack(body_feature,dim=0)
        face_feature  = torch.stack(face_feature,dim=0)
        text_feature  = torch.stack(text_feature,dim=0)
        return place_feature,body_feature,face_feature,text_feature
    def spp(self,ins20):
        ins10 = []
        ins05 = []
        ins02 = []
        ins01 = []
        rel_num,length,dim = ins20.shape
        #10
        for i in range(10):
            start = i*2
            end   = start+2
            ins10.append(ins20[:,start:end,:].mean(dim=1,keepdim=True))
        ins10 = torch.cat(ins10,dim=1)
        #05
        for i in range(5):
            start = i*4
            end   = start+4
            ins05.append(ins20[:,start:end,:].mean(dim=1,keepdim=True))
        ins05 = torch.cat(ins05,dim=1)
        #02
        for i in range(2):
            start = i*10
            end   = start+10
            ins02.append(ins20[:,start:end,:].mean(dim=1,keepdim=True))
        ins02 = torch.cat(ins02,dim=1)
        ins01 = ins20[:,:,:].mean(dim=1,keepdim=True)
        ins = torch.cat([ins20,ins10,ins05,ins02,ins01],dim=1)
        return ins

    def get_topk_tokens(self,tokens,head_score,topk):
        bs,hn,num = head_score.shape
        _,_,dim = tokens.shape
        head_score = head_score.view((bs,hn*num))#b,head,topk
        
        value,sel = torch.topk(head_score,hn*topk, dim=-1)
        selects = []
        for i in range(bs):
            se = []
            for item in sel[i]:
                se.append(item%num)
            temp = []
            [temp.append(i) for i in se if i not in temp]
            s_topk = temp[:topk]

            s_topk.sort()
            selects.append(s_topk)
        tokens_new = torch.zeros([bs,topk,dim])
        for i in range(bs):
            s_topk = selects[i]#topk
            for j in range(topk):
                tokens_new[i][j][:] = tokens[i][s_topk[j]][:]
        tokens_new = tokens_new.to(device)
        return tokens_new

    def process_fine_tokens(self,tokens):
        bs,topk = tokens.shape[0],tokens.shape[1]
        bftp = tokens[:,:,:512*6].clone()#bs,topk,d*6
        va   = tokens[:,0,512*6:].clone().unsqueeze(1)#bs,1,d*2
        bftp = bftp.view((bs,topk*6,512))
        va   = va.view((bs,2,512))
        tokens_new = torch.cat([bftp,va],dim=1)#bs,topk*6+2,d
        return tokens_new

    
    def gen_instances(self,body_feature,face_feature,text_feature,place_feature,video_feature,audio_feature,
                      bodys,faces,texts,pframes,gids,labels,batchsize):
        # face_dim = 512
        # body_dim = 512
        # text_dim = 512
        instances_20 = []
        face_idx = 0
        body_idx = 0
        text_idx = 0
        for idx in range(batchsize):
            persons = []
            label_s = labels[idx]
            for lab in label_s:
                persons.append(lab[0])
                persons.append(lab[2])
            persons = list(set(persons))
            #填充数据
            pfs = place_feature[idx]
            bfs = {}
            ffs = {}
            for p in persons:
                bfs[p] = []
                ffs[p] = []
            tfs = []
            fframes = list(faces[idx].keys())
            bframes = list(bodys[idx].keys())
            tframes = list(texts[idx].keys())
            for i,frame in enumerate(pframes[idx]):
                if frame in fframes:
                    fpersons = list(faces[idx][frame].keys())
                    for p in fpersons:
                        ffs[p].append(face_feature[face_idx])
                        face_idx = face_idx + 1
                    _persons = []
                    for p in persons:
                        if p not in fpersons:
                            _persons.append(p)
                    for p in _persons:
                        ffs[p].append(torch.zeros(512).to(device))
                else:
                    for p in persons:
                        ffs[p].append(torch.zeros(512).to(device))
                if frame in bframes:
                    bpersons = list(bodys[idx][frame].keys())
                    for p in bpersons:
                        bfs[p].append(body_feature[body_idx])
                        body_idx = body_idx + 1

                    _persons = []
                    for p in persons:
                        if p not in bpersons:
                            _persons.append(p)
                    for p in _persons:
                        bfs[p].append(torch.zeros(512).to(device))
                else:
                    for p in persons:
                        bfs[p].append(torch.zeros(512).to(device))
                if frame in tframes:
                    tfs.append(text_feature[text_idx])
                    text_idx = text_idx + 1
                else:
                    tfs.append(torch.zeros(512).to(device))
            feat_s_20 = []
            for lab in label_s:
                p1,p2 = lab[0],lab[2]
                for i,frame in enumerate(pframes[idx]):
                    feat_20 = torch.cat((bfs[p1][i],bfs[p2][i],ffs[p1][i],ffs[p2][i],pfs[i],tfs[i],video_feature[idx],audio_feature[idx]),dim=0)
                    feat_s_20.append(feat_20)
            feat_s_20 = torch.stack(feat_s_20,dim=0)#
            instances_20.append(feat_s_20)
        instances_20 = torch.cat(instances_20,dim=0)
        fshape = feat_s_20.shape
        return instances_20

    def forward(self,faces,bodys,places,texts,audios,videos,labels,pframes,fpersons,bpersons,graphs,gids,batchsize):
        topk_c = configs["topk_c"]
        topk_f = configs["topk_f"]
        face_feature = []
        body_feature = []
        text_feature = []

        face_infos = {}
        body_infos = {}
        text_infos = {}
        face_persons = {}
        body_persons = {}
        text_idx = 0
        face_idx = 0
        body_idx = 0
        sc_frames = []
        for idx in range(batchsize):
            graph_s = graphs[idx]
            face_infos[idx] = {}
            body_infos[idx] = {}
            text_infos[idx] = {}
            face_persons[idx] = []
            body_persons[idx] = []

            for frame in faces[idx]:
                face_infos[idx][frame] = {}
                for person in faces[idx][frame]:
                    face_feature.append(faces[idx][frame][person])
                    face_infos[idx][frame][person] = face_idx
                    face_idx = face_idx+1
            for frame in bodys[idx]:
                body_infos[idx][frame] = {}
                for person in bodys[idx][frame]:
                    body_feature.append(bodys[idx][frame][person])
                    body_infos[idx][frame][person] = body_idx
                    body_idx = body_idx+1
            for i,frame in enumerate(list(texts[idx].keys())):
                text_infos[idx][i] = text_idx
                text_feature.append(texts[idx][frame])
                text_idx = text_idx+1

        face_feature = torch.stack(face_feature,dim=0)
        body_feature = torch.stack(body_feature,dim=0)
        text_feature = torch.stack(text_feature,dim=0)
        
        face_feature = self.face_fc1(face_feature)
        body_feature = self.body_fc1(body_feature)
        text_feature = self.text_fc1(text_feature)
        places = self.place_fc1(places)
        videos = self.video_fc1(videos)
        audios = self.audio_fc1(audios)

        graphs = self.data2graph(places,body_feature,face_feature,text_feature,graphs,pframes,gids,batchsize)

        for idx in range(batchsize):
            for i, graph in enumerate(graphs[idx]):
                graph.ndata["feature"] = self.gcn(graph,graph.ndata["feature"])

        places,body_feature,face_feature,text_feature = self.graph2data(places,bodys,faces,texts,graphs,pframes,batchsize)
        
        face_feature = self.face_fc2(face_feature)
        body_feature = self.body_fc2(body_feature)
        text_feature = self.text_fc2(text_feature)
        places = self.place_fc2(places)
        videos = self.video_fc2(videos)
        audios = self.audio_fc2(audios)
        
        face_feature = self.relu(face_feature)
        body_feature = self.relu(body_feature)
        text_feature = self.relu(text_feature)
        places = self.relu(places)
        videos = self.relu(videos)
        audios = self.relu(audios)
        instances_20 = self.gen_instances(body_feature,face_feature,text_feature,places,videos,audios,
              bodys,faces,texts,pframes,gids,labels,batchsize)
        instances_20_shape = instances_20.shape
        instances_20 = instances_20.view((-1,20,instances_20_shape[-1]))

        instances_21 = self.spp(instances_20)
        ins_21 = instances_21.clone()
        
        #粗粒度分类
        c_tokens = self.insfc(instances_21)
        cls_tokens_c = repeat(self.cls_token_c, '1 1 d -> b 1 d', b = c_tokens.shape[0])
        c_tokens = torch.cat([cls_tokens_c,c_tokens],dim=1)
        
        c_tokens,attention_score_c_1 = self.coarse_attention_1(c_tokens)
        c_tokens,attention_score_c_2 = self.coarse_attention_2(c_tokens)
        cls_tokens_c = c_tokens[:,0,:].unsqueeze(1)
        c_tokens = c_tokens[:,1:,:]
        #score = x[:, :, 0, 1:],IELT，https://github.com/mobulan/IELT/blob/main/models/IELT.py
        head_score_c_1 = attention_score_c_1[:,:,0,1:].clone()
        head_score_c_2 = attention_score_c_2[:,:,0,1:].clone()
        head_score_c = head_score_c_1 * head_score_c_2
        sc_tokens = self.get_topk_tokens(c_tokens,head_score_c,topk=topk_c)
        cfeatures = sc_tokens.clone()
        sc_tokens = torch.cat([cls_tokens_c,sc_tokens],dim=1)
        sc_tokens,_ = self.coarse_attention_class(sc_tokens)
        cls_tokens_c = sc_tokens[:,0,:].unsqueeze(1)
        relations_c  = self.classfier_coarse(cls_tokens_c)
        

        cls_tokens_f = repeat(self.cls_token_f, '1 1 d -> b 1 d', b = ins_21.shape[0])
        cr_tokens = self.get_topk_tokens(ins_21,head_score_c,topk=topk_c)
        fr_tokens = self.process_fine_tokens(cr_tokens)
        fine_tokens = torch.cat([cls_tokens_f,fr_tokens],dim=1)
        fine_tokens,attention_score_f_1 = self.fine_attention_1(fine_tokens)#n,head,len,len
        fine_tokens,attention_score_f_2 = self.fine_attention_2(fine_tokens)#n,head,len,len
        cls_tokens_f = fine_tokens[:,0,:].unsqueeze(1)
        fine_tokens = fine_tokens[:,1:,:]
        head_score_f_1 = attention_score_f_1[:,:,0,1:].clone()
        head_score_f_2 = attention_score_f_2[:,:,0,1:].clone()
        head_score_f = head_score_f_1 * head_score_f_2
        sf_tokens = self.get_topk_tokens(fine_tokens,head_score_f,topk=topk_f)
        ffeatures = sf_tokens.clone()
        
        sf_tokens = torch.cat([cls_tokens_f,sf_tokens],dim=1)
        sf_tokens,_ = self.fine_attention_class(sf_tokens)
        cls_tokens_f = sf_tokens[:,0,:].unsqueeze(1)
        relations_f  = self.classfier_fine(cls_tokens_f)
        
        relations = torch.cat([relations_c,relations_f],dim=1)
        relations,_ = relations.max(1,keepdim=False)

        return relations,cfeatures,ffeatures

class SelfAtt(nn.Module):
    def __init__(self,dim=1024,depth=2):
        super(SelfAtt, self).__init__()
        self.dim=dim
        self.layers = nn.ModuleList([])
        #self.dropouts1 = nn.ModuleList([])
        self.norms1 = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        self.norms2 = nn.ModuleList([])
        #self.dropouts2 = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.MultiheadAttention(embed_dim=self.dim,num_heads=4,dropout=0.1,batch_first=True))
            self.norms1.append(LayerNorm(d_model=self.dim))
            #self.dropouts1.append(nn.Dropout(p=0.1))
            self.ffn_layers.append(PositionwiseFeedForward(self.dim,self.dim*4))
            self.norms2.append(LayerNorm(d_model=self.dim))
            #self.dropouts2.append(nn.Dropout(p=0.1))
    def forward(self,x):
        for att,norm1,ffn,norm2 in zip(self.layers,self.norms1,self.ffn_layers,self.norms2):
            _x = x
            x,_ = att(x,x,x)
            #x = dr1(x)
            x = norm1(x+_x)
            _x = x
            x = ffn(x)
            #x = dr2(x)
            x = norm2(x+_x)
        return x
class CrossAtt(nn.Module):
    def __init__(self,dim=1024,depth=2):
        super(CrossAtt, self).__init__()
        self.dim=dim
        self.layers = nn.ModuleList([])
        #self.dropouts1 = nn.ModuleList([])
        self.norms1 = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        self.norms2 = nn.ModuleList([])
        #self.dropouts2 = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.MultiheadAttention(embed_dim=self.dim,num_heads=4,dropout=0.1,batch_first=True))
            self.norms1.append(LayerNorm(d_model=self.dim))
            #self.dropouts1.append(nn.Dropout(p=0.1))
            self.ffn_layers.append(PositionwiseFeedForward(self.dim,self.dim*4))
            self.norms2.append(LayerNorm(d_model=self.dim))
            #self.dropouts2.append(nn.Dropout(p=0.1))
    def forward(self,x,kv):
        for att,norm1,ffn,norm2 in zip(self.layers,self.norms1,self.ffn_layers,self.norms2):
            _x = x
            x,_ = att(x,kv,kv)
            #x = dr1(x)
            x = norm1(x+_x)
            _x = x
            x = ffn(x)
            #x = dr2(x)
            x = norm2(x+_x)
        return x

class Qformer(nn.Module):
    def __init__(self,dim=1024,depth=2):
        super(Qformer, self).__init__()
        self.dim=dim
        self.query_tokens = nn.Parameter(torch.randn(1, configs["num_query_token"], self.dim))
        self.satts = nn.ModuleList([])
        self.catts = nn.ModuleList([])
        for i in range(configs["layer_num"]):
            self.satts.append(SelfAtt(dim =self.dim))
            self.catts.append(CrossAtt(dim =self.dim))
    def forward(self,features):
        bs = features.shape[0]
        query_tokens = repeat(self.query_tokens, '1 n d -> b n d', b = bs)

        for satt,catt in zip(self.satts,self.catts):
            query_tokens = satt(query_tokens) + query_tokens
            out = catt(query_tokens,features) + query_tokens
        return out

class LEKCS(nn.Module):
    def __init__(self):
        super(LEKCS, self).__init__()

        self.model_gl = Model_GL()

        #qformer
        self.qformer_c = Qformer(dim=1024)
        self.qformer_f = Qformer(dim=512)

        self.cfc = nn.Linear(1024,4096)
        self.ffc = nn.Linear(512,4096)

        self.cls = nn.Sequential(
            nn.Linear(4096,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Linear(128, 8)
        )

    def get_label_list(self,labels):
        label_list = []
        for idx in range(len(labels)):
            temp = labels[idx]
            for rel in temp:
                label_list.append(rel[1])
        return label_list
    def get_sim(self,feature1,label_emb,labels):
        label_list = self.get_label_list(labels)
        bs = feature1.shape[0]
        similarity_matrix = []
        for idx in range(bs):
            x = feature1[idx,:,:]
            sim = F.cosine_similarity(x.unsqueeze(1),label_emb[label_list[idx],:].unsqueeze(0).unsqueeze(0), dim=2)
            similarity_matrix.append(sim.squeeze())
        similarity_matrix = torch.stack(similarity_matrix,dim=0)#bs,length
        return similarity_matrix
        
    def forward(self,faces,bodys,places,texts,audios,videos,labels,pframes,fpersons,bpersons,graphs,gids,batchsize
                ,lab_emb,bos_emb,llmmodel
               ):
        rel,cfeatures,ffeatures = self.model_gl(faces,bodys,places,texts,audios,videos,labels,pframes,fpersons,bpersons,graphs,gids,batchsize)
        queries_c = self.qformer_c(cfeatures)
        queries_f = self.qformer_f(ffeatures)

        queries_c = self.cfc(queries_c)
        queries_f = self.ffc(queries_f)#4096

        cv = 1-self.get_sim(queries_c,lab_emb,labels).mean()
        fv = 1-self.get_sim(queries_f,lab_emb,labels).mean()

        bs = cfeatures.shape[0]
        outputs = []
        for i in range(bs):
            embeds=torch.cat([bos_emb,queries_c[i].unsqueeze(0),queries_f[i].unsqueeze(0)],dim=1)
            with torch.no_grad():
                output=llmmodel(inputs_embeds=embeds,output_hidden_states=True,return_dict=True)
            outputs.append(output)
        outputs = torch.cat(outputs,dim=0)
        hs = outputs.float()[:,-1,:]
        result_llm = self.cls(hs)
        
        result = torch.stack([rel,result_llm],dim=0)
        result,_ = torch.max(result,dim=0,keepdim=False)
        return result,cv,fv      