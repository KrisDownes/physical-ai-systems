"""Evaluator-only candidate rollouts; learned policies never receive simulator truth."""
import copy,gzip,hashlib,json,time
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from shared_tasks import SharedTask
from shared_evaluate import setup,vector,infer
from run_store import create_run

CFG=json.loads(Path('so101/decision_audit_protocol.json').read_text())

def integration_state(e):
    flag=mujoco.mjtState.mjSTATE_INTEGRATION
    x=np.zeros(mujoco.mj_stateSize(e.env.model,flag));mujoco.mj_getState(e.env.model,e.env.data,x,flag);return x

def mirror(source,dest):
    # Copies *all* MjData fields, including solver warmstart, controls and actuator state.
    mujoco.mj_copyData(dest.env.data,dest.env.model,source.env.data)
    dest.last=copy.deepcopy(source.last);dest.last_feature=copy.deepcopy(source.last_feature);dest.offset=source.offset.copy()
    dest.env.action_count=source.env.action_count;dest.env.movement=source.env.movement;dest.env.saturated_actions=source.env.saturated_actions;dest.env.trace=[]
    dest.env.physics_collision_steps=source.env.physics_collision_steps;dest.env.physics_limit_steps=source.env.physics_limit_steps
    dest.audit_time=source.audit_time;dest.physics_warnings=source.physics_warnings.copy();dest.numerical_reset=source.numerical_reset

def score(pos,goal,dy):return abs(float(pos)-goal)+.001*abs(dy)
def rank(scores):
    if any(v is None for v in scores):return None
    return [int(i) for i in np.argsort(scores,kind='stable')]

def audit(source,clone,f,goal,grip,history,weights):
    cand=CFG['candidate_y_m'];before=integration_state(source);statehash=hashlib.sha256(before.tobytes()).hexdigest();before_py=copy.deepcopy((source.last,source.last_feature,source.env.action_count,len(source.env.trace),source.physics_warnings,source.env.saturated_actions))
    ctx=([np.zeros(15,np.float32)]*3+history)[-3:];vals={'current':[],'history':[],'nominal':[],'rollout':[]};outputs=[];predictor_s=0.;rollout_s=0.
    for dy in cand:
        started=time.perf_counter()
        a=np.array([0.,dy]);z=vector(f,a);pc=infer(weights['current'],z)*.01;ph=infer(weights['history'],np.r_[z,*ctx])*.01
        vals['current'].append(score(f[1]+np.clip(pc[1],-.03,.03),goal,dy));vals['history'].append(score(f[1]+np.clip(ph[1],-.03,.03),goal,dy));vals['nominal'].append(score(f[1]+dy,goal,dy))
        predictor_s+=time.perf_counter()-started;started=time.perf_counter()
        mirror(source,clone);delta_before=float(clone.truth()[1]);next_f=clone.step(a,grip);delta_truth=float(clone.truth()[1]-delta_before)
        rollout_s+=time.perf_counter()-started
        vals['rollout'].append(score(next_f[1],goal,dy) if next_f[-1] else None)
        outputs.append({'action_y_m':dy,'current_predicted_y_m':float(pc[1]),'history_predicted_y_m':float(ph[1]),'visual_next_y_m':float(next_f[1]) if next_f[-1] else None,'next_features':next_f.tolist(),'truth_delta_y_evaluator_m':delta_truth,'sent_joint_delta_rad':clone.env.trace[-1]['sent_delta_rad'],'joint_action_saturated':clone.env.saturated_actions>source.env.saturated_actions,'warnings':clone.physics_warnings,'numerical_invalid':clone.numerical_reset,'self_collision_steps':clone.env.physics_collision_steps-source.env.physics_collision_steps,'joint_limit_steps':clone.env.physics_limit_steps-source.env.physics_limit_steps})
    after=integration_state(source);assert np.array_equal(before,after),'diagnostic mutated source integration state'
    assert source.env.action_count==before_py[2] and len(source.env.trace)==before_py[3] and source.physics_warnings==before_py[4]
    assert source.env.saturated_actions==before_py[5]
    assert source.last_feature[0]==before_py[1][0] and np.array_equal(source.last_feature[1],before_py[1][1])
    for key in before_py[0]:assert np.array_equal(source.last[key],before_py[0][key])
    return {'scores_m':vals,'ranking_indices':{k:rank(v) for k,v in vals.items()},'winner_action_y_m':{k:(None if rank(v) is None else cand[rank(v)[0]]) for k,v in vals.items()},'candidates':outputs,'gate_bypasses_planning':bool(abs(float(f[1])-goal)<CFG['goal_gate_m']),'source_integration_sha256':statehash,'source_unchanged':True,'all_predictors_compute_s':predictor_s,'cloned_rollouts_wall_s':rollout_s}

def main():
    run_start=time.perf_counter();out=create_run('decision_audit',CFG);ev=Path(CFG['evaluation_run']);rpath=ev/'results.json'
    if rpath.exists():rows=json.loads(rpath.read_text())
    else:
        with gzip.open(ev/'results.json.gz','rt') as f:rows=json.load(f)
    weights={}
    for name in ('current','history'):
        n=np.load(Path(CFG['training_run'])/f'{name}_seed{CFG["recorded_seed"]}.npz');weights[name]=[n[f'p{i}'] for i in range(4)]
    protocol=json.loads(Path('so101/shared_protocol.json').read_text());records=[];frames=[]
    for task in CFG['recorded_tasks']:
        start=protocol['test_starts'][CFG['recorded_start_index']];row=next(r for r in rows if r['seed']==CFG['recorded_seed'] and r['method']=='history' and r['task']==task and r['start']==start)
        source=SharedTask(task,**row['condition']);f,grip,setting,_=setup(source,task,start,5000+CFG['recorded_start_index']);goal,_=setting
        clone=SharedTask(task,**row['condition']);clone.reset(start['xy'],5000+CFG['recorded_start_index'],start['pan']);hist=[]
        for t in range(max(CFG['decision_steps'])+1):
            expected=row['records'][t];replay_error=float(np.max(np.abs(f-np.asarray(expected['obs']))))
            result=None
            if t in CFG['decision_steps']:
                result=audit(source,clone,f,goal,grip,hist,weights);result.update(task=task,step=t,goal_y=goal,visible_feature=f.tolist(),source_replay_max_error=replay_error,recorded_action=expected['action']);records.append(result);frames.append(source.last['image'].copy())
            action=np.asarray(expected['action']);old=f.copy();f=source.step(action,grip);hist.append(vector(old,action))
            if result is not None:
                idx=min(range(len(CFG['candidate_y_m'])),key=lambda j:abs(CFG['candidate_y_m'][j]-action[1]))
                assert abs(CFG['candidate_y_m'][idx]-action[1])<1e-7
                difference=float(np.max(np.abs(f-np.asarray(result['candidates'][idx]['next_features']))))
                result['cloned_vs_actual_next_feature_max_error']=difference
                assert difference<1e-6,'cloned rollout does not match actual continuation'
        source.close();clone.close()
    pair=[]
    for cond in CFG['condition_pair']:
        start=CFG['paired_start'];src=SharedTask('pull',**cond);f,grip,setting,truth=setup(src,'pull',start,CFG['paired_seed']);goal,_=setting
        clone=SharedTask('pull',**cond);clone.reset(start['xy'],CFG['paired_seed'],start['pan']);hist=[];stages=[]
        for t in range(len(CFG['paired_actions'])+1):
            a=audit(src,clone,f,goal,grip,hist,weights);a.update(step=t,visible_feature=f.tolist(),goal_y=goal,truth_position_evaluator=src.truth().tolist());stages.append(a)
            if t<len(CFG['paired_actions']):
                action=np.array([0.,CFG['paired_actions'][t]]);old=f.copy();f=src.step(action,grip);hist.append(vector(old,action))
        pair.append({'condition_evaluator':cond,'stages':stages});src.close();clone.close()
    # Evaluator-only counterfactual: exactly the same complete dynamic state,
    # scored in two models differing only in slider resistance. The common
    # prehistory was generated by the first model; this is not a deployment test.
    start=CFG['paired_start'];low=SharedTask('pull',**CFG['condition_pair'][0]);f,grip,setting,_=setup(low,'pull',start,CFG['paired_seed']);goal,_=setting
    high=SharedTask('pull',**CFG['condition_pair'][1]);high.reset(start['xy'],CFG['paired_seed'],start['pan'])
    lowclone=SharedTask('pull',**CFG['condition_pair'][0]);lowclone.reset(start['xy'],CFG['paired_seed'],start['pan'])
    highclone=SharedTask('pull',**CFG['condition_pair'][1]);highclone.reset(start['xy'],CFG['paired_seed'],start['pan'])
    exact=[];hist=[]
    for t in range(len(CFG['paired_actions'])+1):
        mirror(low,high);assert np.array_equal(integration_state(low),integration_state(high))
        l=audit(low,lowclone,f,goal,grip,hist,weights);h=audit(high,highclone,f,goal,grip,hist,weights)
        exact.append({'step':t,'low':l,'high':h,'state_vectors_equal':True,'visible_feature':f.tolist()})
        if t<len(CFG['paired_actions']):
            action=np.array([0.,CFG['paired_actions'][t]]);old=f.copy();f=low.step(action,grip);hist.append(vector(old,action))
    for env in (low,high,lowclone,highclone):env.close()
    result={'recorded_decisions':records,'natural_pair':pair,'exact_state_counterfactual':exact,'wall_s_before_video':time.perf_counter()-run_start};(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if frames:imageio.mimsave(out/'recorded_states.gif',frames,duration=600,loop=0)
    print('saved',out)
    for r in records:print(r['task'],r['step'],'gate',r['gate_bypasses_planning'],'winner',r['winner_action_y_m'],'replay_error',r['source_replay_max_error'],flush=True)
    for pairrow in pair:
        for stage in pairrow['stages']:print('resistance',pairrow['condition_evaluator']['resistance'],'step',stage['step'],'winner',stage['winner_action_y_m'],'truth',stage['truth_position_evaluator'][1],flush=True)
if __name__=='__main__':main()
