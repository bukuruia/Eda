"""Load152 pinned public test members;149 have directly supplied public OOF."""
import ast,json
from pathlib import Path
import numpy as np,pandas as pd
from nov21_board import L
from nov16_direct_versions import ROOT,sha,write

R=ROOT/'public/nov21_round4';I=ROOT/'private_lab/local_validator/inputs'

class PublicPool:
    def __init__(self):
        source=ROOT/'public/nov21_round3/good/s6e11-board-model-realmlp-lb-0-95540.ipynb'
        code='\n'.join(''.join(c['source']) for c in json.loads(source.read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code')
        n=next(n for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STEP_Q' for t in n.targets))
        self.names=list(json.loads(ast.literal_eval(n.value.args[0])))
        self.z=np.load(I/'q17/rank1_q17_streams.npz',allow_pickle=False);self.y=self.z['y_train'].astype(int);self.folds=self.z['folds']
        assert sha(I/'q17/rank1_q17_streams.npz')=='0fe4ab1263a673c8065d7145590b92da440263161bf860ce45f2976b5c4fa56f'
        self.b33_names=list(pd.read_csv(R/'b33/manifest.csv').name)
        self.b33={p:np.load(R/'b33'/f'{p}.npy',mmap_mode='r') for p in ['test','oof']}
        sums={line.split()[1]:line.split()[0] for line in (R/'pins/SHA256SUMS').read_text().splitlines()}
        for name,digest in sums.items():
            p=R/'pins'/name
            if p.exists():assert sha(p)==digest,(name,sha(p))
        self.ids={p:pd.read_csv(ROOT/'data'/('test.csv' if p=='test' else 'train.csv'),usecols=['id']).id.to_numpy() for p in ['test','oof']}
        self.sources={};self.csv_cache={}
        self.actual_oof_names=[k for k in self.names if k not in ['x:Sold','x:rmlpnew','x:rmlpold']]
        assert len(self.names)==152 and len(self.actual_oof_names)==149

    def csv(self,path,part,column):
        if str(path) not in self.csv_cache:
            f=pd.read_csv(path);self.csv_cache[str(path)]=f.set_index('id').loc[self.ids[part],column].to_numpy()
        return self.csv_cache[str(path)]

    def values(self,key,part):
        if key.startswith('s:'):return self.z[part+'_'+key[2:]]
        if key.startswith('b:'):return np.load(I/'oof-bank'/f'{part}_{key[2:]}.npy')
        if key.startswith('x:b33_'):return self.b33[part][:,self.b33_names.index(key[6:])]
        if key=='wide':
            path=ROOT/'public/nov08/wide'/('wide_lm_stack_test.npy' if part=='test' else 'wide_lm_stack_nested.npy')
            expected='b37471e6c34a50715284e807d68e15e5f3ad033e72400904f27876b24041f8cc' if part=='test' else '69375871698f1adf2b4c666fc7f5180edf00ac6836668353b8e7e4ca43eb2722'
            assert sha(path)==expected;return np.load(path)
        if key.startswith('x:weni_'):return np.load(R/'pins'/f'{part}_weni_{key[7:]}.npy')
        if key=='x:tabm':return np.load(R/'b33'/f'tabm_{part}.npy')
        if key=='x:davut':return self.csv(R/'pins'/f'davutkursun_xgb_glm_bag_{part}.csv',part,'pred')
        if key=='x:yek_tabm':return self.csv(R/'pins'/('yekenot_tabm_pytabkit_submission.csv' if part=='test' else 'yekenot_tabm_pytabkit_oof_preds.csv'),part,'is_fraud')
        files={'x:Sold':'alexchilton_v23_submission.csv','x:rmlpnew':'yekenot_realmlp_pytabkit_v15_submission.csv','x:rmlpold':'yekenot_realmlp_pytorch_v28_submission.csv'}
        assert part=='test',('No exact OOF for this member',key)
        path=R/'streams'/files[key]
        expected={'x:Sold':'36686835b6ca3bd805ef578fc50cd62b3cbbe4d2a70275e50d6c38c072d4ad77','x:rmlpnew':'b1462d0c7f64a48a4739acc4e68673d63e4648660744acb734b6770bc5207055','x:rmlpold':'c8ff11aeedb8e7a6268a1617ae14133f2dc7fa347a581f3cceed0ac6d44bfc98'}[key]
        assert sha(path)==expected
        return self.csv(path,part,'is_fraud')

    def test_matrix(self):
        return np.stack([L(self.values(k,'test')) for k in self.names]).astype(np.float32)

    def oof_matrix(self):
        x=np.empty((len(self.actual_oof_names),len(self.y)),np.float32)
        for j,k in enumerate(self.actual_oof_names):
            raw=self.values(k,'oof')
            assert raw.shape==self.y.shape and np.isfinite(raw).all()
            for f in range(5):
                mask=self.folds==f;x[j,mask]=L(raw[mask])
        return x
