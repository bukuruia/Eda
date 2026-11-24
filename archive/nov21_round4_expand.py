"""Rebuild additional scored public recipes and pin their rank vectors."""
import io,json,hashlib
import numpy as np,pandas as pd
from nov21_board import rank,L
from nov16_direct_versions import ROOT,sha,write

OUT=ROOT/'private_lab/nov21_round4_expanded';OUT.mkdir(exist_ok=True)
R=ROOT/'public/nov21_round4';PREV=ROOT/'private_lab/nov21_round3_board'


def main():
    sample=pd.read_csv(ROOT/'data/sample_submission.csv');old=json.loads((PREV/'input_manifest.json').read_text());manifest=[];arrays={};seen={}
    def add(key,score,path,stage,source,recipe,exact=True):
        f=pd.read_csv(path);assert f.id.equals(sample.id) and np.isfinite(f.is_fraud).all();u=rank(f.is_fraud.to_numpy());digest=hashlib.sha256(u.tobytes()).hexdigest()
        if digest in seen:
            assert seen[digest]['score']==score,(key,seen[digest]);return
        record=dict(key=key,score=score,path=str(path),sha256=sha(path),rank_sha256=digest,stage=stage,source=source,recipe=recipe,exact_original_artifact=exact,extra_score_variance_points=0. if exact else .25**2)
        seen[digest]=record;manifest.append(record);arrays[key]=u
    for r in old:add(r['key'],r['score'],r['path'],r['stage'],'Previously pinned public observations','See previous manifest')
    add('own_v38',.95540,ROOT/'private_lab/nov21_round3_live/round3_1/cloud/submission.csv',33,'malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion','Executedv38')
    def U(v):
        buff=io.StringIO();sample.assign(is_fraud=rank(v)).to_csv(buff,index=False);buff.seek(0);return pd.read_csv(buff).is_fraud.to_numpy()
    def make(key,score,values,stage,recipe,source='goodpjw2008/s6e11-board-model-realmlp-lb-0-95540'):
        p=OUT/(key+'.csv');sample.assign(is_fraud=rank(values)).to_csv(p,index=False,lineterminator='\n');add(key,score,p,stage,source,recipe,exact=False);return p
    def stream(filename,expected):
        p=R/'streams'/filename;assert sha(p)==expected;f=pd.read_csv(p);assert f.id.equals(sample.id);return f.is_fraud.to_numpy()
    G=stream('goodpjw2008_v9_submission.csv','4d88b6f6dc7612df8d5ff9d77799748815b79f9762103341c0be48aa3dd45c1d')
    B=stream('bhuvannr_step10_submission.csv','ce020d84c6279f5946f94983db6fe68b96a0dd96b5809d0a96ecd263fe586703')
    new=stream('yekenot_realmlp_pytabkit_v15_submission.csv','b1462d0c7f64a48a4739acc4e68673d63e4648660744acb734b6770bc5207055')
    previous=stream('yekenot_realmlp_pytorch_v28_submission.csv','c8ff11aeedb8e7a6268a1617ae14133f2dc7fa347a581f3cceed0ac6d44bfc98')
    # Busy PRIME uses np.log(r/(1-r)), not scipy.special.logit. Their last
    # bits can change a tied pair before averaging ranks; preserve the writer.
    bL=lambda v:np.log(rank(v)/(1-rank(v)))
    d=bL(new)-bL(previous);tabm=np.load(R/'b33/tabm_test.npy');td=bL(tabm)-bL(previous)
    proofs={};busy='busyaprime/s6e11-124-voices-the-top-rebuilt-lb-0-95465'
    def recipe(w,t):return .5*(U(bL(G)+w*d+t*td)+U(bL(B)+w*d+t*td))
    p=make('busy_base042',.95442,recipe(.4,0),4.4,'U(mean(U(L(G)+.4D),U(L(B)+.4D)))',busy)
    assert sha(p)=='46eb96d06ea474a16029ee65df7fdbfcf44ee1b6197e8b2e0136ab908f5a6299';proofs['busy_base042_exact']=True
    record=next((r for r in manifest if r['key']=='busy_base042'),None)
    if record:record.update(exact_original_artifact=True,extra_score_variance_points=0.)
    for w,score in [(.2,.95439),(.5,.95441),(.6,.95440)]:make('rmlp_curve_'+str(w),score,recipe(w,0),4.45+w*.01,f'Same two-branch recipe, RealMLP step{w}',busy)
    st=None
    for w,score in [(.03,.95450),(.05,.95455),(.07,.95459),(.1,.95463),(.15,.95465),(.2,.95463)]:
        if w in [.05,.15]:
            p=R/'busy/output'/('submission.csv' if w==.15 else 'candidate_04_D_+_005_TabM.csv')
            expected='bc386aa53bc360de33e5f6b5f9b16d046406adc1e7d3f4a0f6db33a697aa6244' if w==.15 else '7446403abd5b3675d5b8da2f6d3dfc2651c073f67ec8397251907940377b2d4d'
            assert sha(p)==expected;add('tabm_curve_'+str(w),score,p,4.6+w*.01,busy,'Original measured artifact',exact=True)
            if w==.15:st=pd.read_csv(p).is_fraud.to_numpy();proofs['busy_tabm015_exact']=True
        else:p=make('tabm_curve_'+str(w),score,recipe(.4,w),4.6+w*.01,f'Same two-branch recipe; reported TabM{w} score, platform-rounding identity not verified',busy)
    # Scored one-dimensional extensions, explicitly reported by the public author.
    for factor,score in [(1.5,.95452),(2.,.95452)]:
        make('C_times_'+str(factor),score,L(arrays['v17'])+factor*(L(arrays['C'])-L(arrays['v17'])),4.1+factor*.001,'Step C extension from measured B1.5')
    c15=pd.read_csv(OUT/'C_times_1.5.csv').is_fraud.to_numpy()
    calc=rank(L(c15)+L(st)-L(recipe(.4,0)));proofs['v19_rank_equal']=bool(np.array_equal(calc,arrays['v19']))
    # Even if intermediary CSV rounding changes a few tied rows, the primitive
    # curve files above are SHA-identical to the scored source files.
    for factor,score in [(1.5,.95483),(2.,.95483),(3.,.95481)]:
        make('D_times_'+str(factor),score,L(arrays['v19'])+factor*(L(arrays['D'])-L(arrays['v19'])),6.1+factor*.001,'Step D extension from measured v19')
    make('E_times_2',.95486,L(arrays['v20'])+2*(L(arrays['E'])-L(arrays['v20'])),8.1,'Step E extension from measured v20')
    for factor,score in [(2.,.95491),(3.,.95493),(4.5,.95495),(6.,.95496),(8.,.95495),(10.,.95492)]:
        p=make('F_times_'+str(factor),score,L(arrays['E'])+factor*(L(arrays['F'])-L(arrays['E'])),9.1+factor*.001,'Step F extension from measured E')
        if factor==6:
            proofs['F6_rank_equal']=bool(np.array_equal(rank(pd.read_csv(p).is_fraud),arrays['v22']))
    for factor,score in [(1.25,.95527),(1.5,.95528),(2.,.95528)]:
        make('L_times_'+str(factor),score,L(arrays['own_v32'])+factor*(L(arrays['L'])-L(arrays['own_v32'])),19.1+factor*.001,'Step L extension from measured ownv32')
    names=[('a_published_residual100.csv',.95534),('b_areeb_zones.csv',.95534),('c_areeb_published.csv',.95534),('d_good_n.csv',.95533),('e_prior_step_orthogonal.csv',.95534),('f_residual_step_orthogonal.csv',.95534),('g_quarter_step_zones.csv',.95533),('h_fusion15_published.csv',.95533),('i_good_m.csv',.95531),('j_equal_best_references.csv',.95534)]
    for i,(name,score) in enumerate(names):add('sriramN_'+str(i),score,R/'sriramN/output'/name,26.8+i*.001,'sriram00777/s6e11-rank-residuals-eda-lb-0-95534','Artifact and measured score identified by source table')
    manifest.sort(key=lambda r:r['stage']);arrays={r['key']:arrays[r['key']] for r in manifest}
    write(OUT/'input_manifest.json',manifest);write(OUT/'identity_proofs.json',proofs);np.savez_compressed(OUT/'scored_rankings.npz',id=sample.id.to_numpy(),**arrays)
    print('EXPANDED PUBLIC RECORD',len(manifest),json.dumps(proofs),flush=True)


if __name__=='__main__':main()
