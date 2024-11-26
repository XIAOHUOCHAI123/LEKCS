from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,face_features,body_features,place_features,text_features,audio_features,video_features,
                 labels,pframes,fpersons,bpersons,gs,gids):
        self.face_features  = face_features
        self.body_features  = body_features
        self.place_features = place_features
        self.text_features  = text_features
        self.audio_features = audio_features
        self.video_features = video_features
        self.labels         = labels
        self.pframes        = pframes
        self.fpersons       = fpersons
        self.bpersons       = bpersons
        self.graphs         = gs
        self.gids           = gids
    def __getitem__(self,index):
#         print(index)
        face_feature  = self.face_features[index]
        body_feature  = self.body_features[index]
        place_feature = self.place_features[index]
        text_feature  = self.text_features[index]
        audio_feature = self.audio_features[index]
        video_feature = self.video_features[index]
        label         = self.labels[index]
        pframe        = self.pframes[index]
        fperson       = self.fpersons[index]
        bperson       = self.bpersons[index]
        g             = self.graphs[index]
        gid           = self.gids[index]

        return face_feature,body_feature,place_feature,text_feature,audio_feature,video_feature,label,pframe,fperson,bperson,g,gid
    def __len__(self):
        return len(self.labels)