import torch
import torch.nn as nn
import numpy as np
from transformer.layer_norm import LayerNorm
from transformer.multi_head_attention import MultiHeadAttention
from transformer.position_wise_feed_forward import PositionwiseFeedForward
from einops import repeat
from llama.modeling_llama import LlamaForSequenceClassification
#To accelerate training, keep main characters in the training set
class Model_Select(nn.Module):
    def __init__(self,cdim,fdim):
        super(Model_Select, self).__init__()
        self.cdim = cdim
        self.fdim = fdim
        self.cls_token_c = nn.Parameter(torch.randn(1, 1, cdim))
        self.cls_token_f = nn.Parameter(torch.randn(1, 1, fdim))
        self.coarse_attention_1 = EncoderLayer(d_model=cdim, ffn_hidden=cdim, n_head=2, drop_prob=0.2)
        self.coarse_attention_2 = EncoderLayer(d_model=cdim, ffn_hidden=cdim, n_head=2, drop_prob=0.2)
        self.coarse_attention_class = EncoderLayer(d_model=cdim, ffn_hidden=cdim, n_head=2, drop_prob=0.2)
        self.fine_attention_1 = EncoderLayer(d_model=fdim, ffn_hidden=fdim, n_head=2, drop_prob=0.2)
        self.fine_attention_2 = EncoderLayer(d_model=fdim, ffn_hidden=fdim, n_head=2, drop_prob=0.2)
        self.fine_attention_class = EncoderLayer(d_model=fdim, ffn_hidden=fdim, n_head=2, drop_prob=0.2)
        self.classfier_coarse = nn.Linear(cdim,8)
        self.classfier_fine   = nn.Linear(fdim,8)
    def get_topk_tokens(self,tokens,head_score,topk):
        bs,hn,num = head_score.shape
        _,_,dim = tokens.shape
        head_score = head_score.view((bs,hn*num))
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
            s_topk = selects[i]
            for j in range(topk):
                tokens_new[i][j][:] = tokens[i][s_topk[j]][:]
        tokens_new = tokens_new.to(device)
        return tokens_new
    def process_fine_tokens(self,tokens):
        bs,topk = tokens.shape[0],tokens.shape[1]
        bftp = tokens[:,:,:self.fdim*6].clone()
        va   = tokens[:,0,self.fdim*6:].clone().unsqueeze(1)
        bftp = bftp.view((bs,topk*6,self.fdim))
        va   = va.view((bs,2,self.fdim))
        tokens_new = torch.cat([bftp,va],dim=1)
        return tokens_new
    def forward(self,c_tokens,f_tokens):
        topk_c = configs["topk_c"]
        topk_f = configs["topk_f"]
        cls_tokens_c = repeat(self.cls_token_c, '1 1 d -> b 1 d', b = c_tokens.shape[0])
        c_tokens = torch.cat([cls_tokens_c,c_tokens],dim=1)
        c_tokens,attention_score_c_1 = self.coarse_attention_1(c_tokens)
        c_tokens,attention_score_c_2 = self.coarse_attention_2(c_tokens)
        cls_tokens_c = c_tokens[:,0,:].unsqueeze(1)
        c_tokens = c_tokens[:,1:,:]
        head_score_c_1 = attention_score_c_1[:,:,0,1:].clone()
        head_score_c_2 = attention_score_c_2[:,:,0,1:].clone()
        head_score_c = head_score_c_1 * head_score_c_2
        sc_tokens = self.get_topk_tokens(c_tokens,head_score_c,topk=topk_c)
        cfeatures = sc_tokens.clone()
        sc_tokens = torch.cat([cls_tokens_c,sc_tokens],dim=1)
        sc_tokens,_ = self.coarse_attention_class(sc_tokens)
        cls_tokens_c = sc_tokens[:,0,:].unsqueeze(1)
        relations_c  = self.classfier_coarse(cls_tokens_c)
        
        cls_tokens_f = repeat(self.cls_token_f, '1 1 d -> b 1 d', b = f_tokens.shape[0])
        cr_tokens = self.get_topk_tokens(f_tokens,head_score_c,topk=topk_c)
        fr_tokens = self.process_fine_tokens(cr_tokens)
        fine_tokens = torch.cat([cls_tokens_f,fr_tokens],dim=1)
        fine_tokens,attention_score_f_1 = self.fine_attention_1(fine_tokens)
        fine_tokens,attention_score_f_2 = self.fine_attention_2(fine_tokens)
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
class LLM(nn.Module):
    def __init__(self):
        super(LLM, self,cdim,fdim).__init__()
        self.model_gl = Model_GL()
        self.qformer_c = Qformer(dim=cdim)
        self.qformer_f = Qformer(dim=fdim)
        self.cfc = nn.Linear(cdim,4096)
        self.ffc = nn.Linear(fdim,4096)
        self.cls = nn.Sequential(
            nn.Linear(4096,1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 8)
        )
    def get_sim(self,feature1,label_emb,label_list):
        bs = feature1.shape[0]
        similarity_matrix = []
        for idx in range(bs):
            x = feature1[idx,:,:]
            sim = F.cosine_similarity(x.unsqueeze(1),label_emb[label_list[idx],:].unsqueeze(0).unsqueeze(0), dim=2)
            similarity_matrix.append(sim.squeeze())
        similarity_matrix = torch.stack(similarity_matrix,dim=0)
        return similarity_matrix 
    def forward(self,label_list,lab_emb,bos_emb,llmmodel,rel,cfeatures,ffeatures):
        queries_c = self.qformer_c(cfeatures)
        queries_f = self.qformer_f(ffeatures)
        queries_c = self.cfc(queries_c)
        queries_f = self.ffc(queries_f)
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