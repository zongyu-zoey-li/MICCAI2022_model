# code that can be useful for data preparation
import pandas as pd
import pdb
import os
all_g =[]
for sub in raw_feature_dir:
    ges_dir_all=glob.glob(os.path.join("/".join(sub.split('/')[0:-2]),"transcriptions/*"))
    kine_dir_all = glob.glob(os.path.join(sub,'*'))
    for ges_dir in ges_dir_all:
        tg=pd.read_csv(ges_dir,sep="\s+",header=None)
        out = filter(lambda s:ges_dir.split('/')[-1].split('_')[-1] in s,kine_dir_all)
        kin_dir = list(out)
        if len(kin_dir)==0:continue
        tb=pd.read_csv(kin_dir[0],sep="\s+",header=None)
        for i in range(tg.shape[0]):
            start_ = tg.iloc[i,0]
            end_ = tg.iloc[i,1]
            gesture = tg.iloc[i,2]
            fill = [gesture]*int(end_-start_+1)
            tb.loc[start_-1:end_-1,'Y']=fill
        if not os.path.exists(os.path.join("/".join(sub.split('/')[0:-2]),'kin_ges')):
           os.makedirs(os.path.join("/".join(sub.split('/')[0:-2]),'kin_ges'),exist_ok=True)
        save_dir = os.path.join(os.path.join("/".join(sub.split('/')[0:-2]),'kin_ges'),kin_dir[0].split('/')[-1])
        #breakpoint()
        tb.to_csv(save_dir,index=None)

###########################################################3
# code for making the y transformation model categorical-> ordinal eg('G1'-> 1, 'G2'->2)

all_g =[]
for sub in raw_feature_dir:
    paths = glob.glob(os.path.join(sub,'*'))
    for p in paths:
        tb = pd.read_csv(p)
        all_g.extend(tb['Y'])


all_g = np.array(all_g)
g_total=[g for g in all_g if g!='nan' ]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(g_total)
#z=le.transform(list(g_total))

import pickle
with open('JIGSAWS-TRANSFORM.pkl','wb') as f:
    pickle.dump(le,f)