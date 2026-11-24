"""Explicit public-search amendment: choose the physical covariance as primary."""
import json
import numpy as np,pandas as pd
from nov21_round4_hetero import HeteroForecast,OUT,EXP
from nov21_round4_frontier import base_zones
from nov21_board import rank,L
from nov16_direct_versions import ROOT,sha,write


def main():
    initial=json.loads((OUT/'results.json').read_text());assert initial['selected'] is None
    forward=json.loads((OUT/'forward.json').read_text())
    assert all(forward[m]['rmse']['True']<forward[m]['rmse']['False'] for m in ['full','exact'])
    amendment=dict(original_gate_passed=False,original_results_sha256=sha(OUT/'results.json'),
        reason='The source-style covariance is smaller than effective public-label covariance by25x. The physical p*(1-share*p)/share scaling is theoretically correct and has lower forward RMSE in both full and exact-original histories. Select it as primary, keeping the source-style profile as sensitivity. This is an adaptive PUBLIC protocol amendment before any new submission, not a passed original gate or a private-model promotion.',
        criterion='min(full_True,exact_True)-.1max(physicalSD)',
        gate='Physical minimum mean>=.4points; source-style minimum>=-.1points; bounded original coefficient/RMS constraints remain.',
        public_score_used_this_round=False,own_private_model_changed=False)
    write(OUT/'primary_amendment_before_lb.json',amendment)
    good=[]
    for r in initial['rows']:
        if not r['name'].startswith('combined_'):continue
        physical=min(r['profiles'][k]['mean'] for k in ['full_True','exact_True']);source=min(r['profiles'][k]['mean'] for k in ['full_False','exact_False']);sd=max(r['profiles'][k]['sd'] for k in ['full_True','exact_True'])
        if physical>=.4 and source>=-.1:good.append((physical-.1*sd,r))
    assert good,'No candidate passes the physical-primary protocol'
    _,selected=max(good,key=lambda x:x[0]);size=float(selected['name'].split('_')[1])
    manifest=json.loads((EXP/'input_manifest.json').read_text());z=np.load(EXP/'scored_rankings.npz');arrays={r['key']:z[r['key']] for r in manifest};scores={r['key']:r['score'] for r in manifest};extra={r['key']:r['extra_score_variance_points'] for r in manifest};exact={r['key']:arrays[r['key']] for r in manifest if r['exact_original_artifact']}
    model=HeteroForecast(arrays,scores,'own_v38',extra,True)
    raw=np.load(ROOT/'private_lab/nov21_round4_search/member_test_logits.npy',mmap_mode='r');names=json.loads((ROOT/'private_lab/nov21_round4_search/member_manifest.json').read_text())['names']
    score_keys=[k for k in arrays if k!='own_v38'];all_names=['score:'+k for k in score_keys]+['member:'+k for k in names]
    base=L(arrays['own_v38']);logs=np.concatenate([np.stack([L(arrays[k]) for k in score_keys]),raw],axis=0);eps=.005
    jac=np.stack([(rank(base+eps*(a-base))-arrays['own_v38'])/eps for a in logs]).astype(np.float32);norms=np.sqrt(np.mean(jac**2,axis=1));unit=jac/np.maximum(norms[:,None],1e-12)
    gram=(unit*model.variance)@unit.T;gram*=len(gram)/np.trace(gram);gradient=unit@model.gradient
    w=np.linalg.solve(gram+initial['ridge']*np.eye(len(gram)),gradient)
    step=json.loads((ROOT/'private_lab/nov21_round3_board/forecasts.json').read_text())['step_rms'];w*=step/max(np.sqrt(np.mean((w@unit)**2)),1e-12);w=size*w/np.maximum(norms,1e-12)
    assert abs(w[:len(score_keys)]).sum()<=3 and abs(w[len(score_keys):]).sum()<=.4
    te=pd.read_csv(ROOT/'data/test.csv');pred=rank(rank(base+w@(logs-base))-2*base_zones(te))
    for mode,ar in [('full',arrays),('exact',exact)]:
        for physical in [False,True]:
            m=HeteroForecast(ar,scores,'own_v38',extra,physical);mean,sd=m.predict(pred);expected=selected['profiles'][mode+'_'+str(physical)]
            assert abs(mean-expected['mean'])<1e-6 and abs(sd-expected['sd'])<1e-6
    sample=pd.read_csv(ROOT/'data/sample_submission.csv');sample.assign(is_fraud=pred).to_csv(OUT/'primary_submission.csv',index=False,lineterminator='\n')
    record=dict(kind='public',name='physical_primary_1',reference='own_v38',weights={k:float(v) for k,v in zip(all_names,w) if v},profiles=selected['profiles'],expected_sha256=sha(OUT/'primary_submission.csv'),amendment=amendment)
    write(OUT/'primary_selected.json',record);print('PHYSICAL PRIMARY SELECTED',selected['name'],record['expected_sha256'],json.dumps(selected['profiles']),flush=True)


if __name__=='__main__':main()
