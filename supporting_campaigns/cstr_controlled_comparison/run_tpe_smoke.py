#!/usr/bin/env python3
"""NON_PUBLICATION_SMOKE_TEST: five-trial Optuna TPE bridge runner."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import optuna

ROOT = Path(__file__).resolve().parent
SPACE_PATH = ROOT / "search_space_cstr_reduced.json"
MATLAB = Path("/Applications/MATLAB_R2026a.app/bin/matlab")
SEED = 20260806
STARTUP = 2
REQUIRED = {"execution_success","metrics","s","v","q","accepted","incumbent_before","incumbent_after","incumbent_updated"}

def load_space():
    space=json.loads(SPACE_PATH.read_text(encoding="utf-8"))
    names=[x["name"] for x in space["variables"]]
    expected=["horizon_max","Wy_base","Wy_gain","Wu","Wdu_base","Wdu_gain","hessian_reg"]
    if names != expected: raise RuntimeError("Canonical seven-variable schema mismatch")
    return space

def build_study(space, seed=SEED, n_startup=STARTUP):
    sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=n_startup,multivariate=True)
    return optuna.create_study(direction="maximize",sampler=sampler)

def suggest_parameters(trial, space):
    values={}
    for spec in space["variables"]:
        n=spec["name"]
        if spec["type"] == "integer": values[n]=trial.suggest_int(n,int(spec["lo"]),int(spec["hi"]))
        elif spec["sampling"] == "log_uniform": values[n]=trial.suggest_float(n,float(spec["lo"]),float(spec["hi"]),log=True)
        else: values[n]=trial.suggest_float(n,float(spec["lo"]),float(spec["hi"]))
    return values

def candidate_hash(params):
    names=["horizon_max","Wy_base","Wy_gain","Wu","Wdu_base","Wdu_gain","hessian_reg"]
    payload="|".join(f"{n}={float(params[n]):.17g}" for n in names)
    return hashlib.sha256(payload.encode()).hexdigest()

def optimizer_objective(result):
    return float(result["q"] if result["execution_success"] and result["accepted"] else result["q"]-1.0)

def replay_proposals(space, objectives, seed=SEED, n_startup=STARTUP):
    study=build_study(space,seed,n_startup); rows=[]
    for objective in objectives:
        trial=study.ask(); params=suggest_parameters(trial,space); h=candidate_hash(params)
        study.tell(trial,float(objective)); rows.append({"trial":trial.number,"params":params,"candidate_id":h})
    return rows

def matlab_call(request):
    with tempfile.TemporaryDirectory(prefix="cstr_tpe_bridge_") as td:
        inp=Path(td)/"input.json"; out=Path(td)/"output.json"; inp.write_text(json.dumps(request),encoding="utf-8")
        matlab_expr=f"evaluate_tpe_trial('{str(inp).replace(chr(39),chr(39)*2)}','{str(out).replace(chr(39),chr(39)*2)}')"
        proc=subprocess.run([str(MATLAB),"-batch",f"addpath('{str(ROOT).replace(chr(39),chr(39)*2)}'); {matlab_expr}"],text=True,capture_output=True)
        if proc.returncode != 0 or not out.exists(): raise RuntimeError(f"MATLAB bridge failed: {proc.stderr[-2000:]}")
        return json.loads(out.read_text(encoding="utf-8"))

def validate_result(result):
    missing=sorted(REQUIRED-set(result))
    if missing: raise ValueError("Missing MATLAB result fields: "+", ".join(missing))

def write_json(path,data): path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=True),encoding="utf-8")

def run(seed=SEED,trials=5,n_startup=STARTUP):
    if trials != 5: raise ValueError("TPE smoke test requires exactly five trials")
    outdir=ROOT/"NON_PUBLICATION_SMOKE_TEST"/f"tpe_seed_{seed}"
    if outdir.exists() and any(outdir.iterdir()): raise FileExistsError(f"Refusing to overwrite non-empty output: {outdir}")
    outdir.mkdir(parents=True,exist_ok=True)
    space=load_space(); study=build_study(space,seed,n_startup); run_id=f"tpe_seed_{seed}"; timer=time.perf_counter()
    manifest={"schema_version":"NON_PUBLICATION_SMOKE_TEST_v1","run_id":run_id,"method":"tpe","smoke_test":True,"publication_eligible":False,"python_executable":sys.executable,"python_version":sys.version,"optuna_version":optuna.__version__,"virtual_environment":str(ROOT/'.venv'),"sampler":{"class":"TPESampler","seed":seed,"n_startup_trials":n_startup,"multivariate":True,"pruning":False},"search_space":str(SPACE_PATH),"matlab_path":str(MATLAB)}
    write_json(outdir/"run_manifest.json",manifest)
    baseline=matlab_call({"mode":"baseline"}); validate_result(baseline); write_json(outdir/"baseline.json",baseline)
    incumbent_q=float(baseline["q"]); incumbent_id=baseline["candidate_id"]; records=[]; accepted=failed=updates=0
    with (outdir/"evaluations.jsonl").open("w",encoding="utf-8") as log:
        for number in range(5):
            trial=study.ask(); params=suggest_parameters(trial,space); cid=candidate_hash(params)
            before=incumbent_q
            try: result=matlab_call({"mode":"trial","sampled_parameters":params,"incumbent_before":before,"incumbent_candidate_id":incumbent_id}); validate_result(result)
            except Exception as exc:
                result={"execution_success":False,"metrics":[],"s":float('-inf'),"v":100.0,"q":float('-inf'),"accepted":False,"incumbent_before":before,"incumbent_after":before,"incumbent_updated":False,"exception_message":str(exc)}
            objective=optimizer_objective(result); study.tell(trial,objective)
            if result.get("incumbent_updated"):
                incumbent_q=float(result["incumbent_after"]); incumbent_id=cid; updates+=1
            accepted+=int(bool(result["accepted"])); failed+=int(not result["execution_success"])
            rec={"schema_version":"NON_PUBLICATION_SMOKE_TEST_v1","run_id":run_id,"method":"tpe","smoke_test":True,"publication_eligible":False,"trial_number":number,"tpe_seed":seed,"optuna_version":optuna.__version__,"startup_trial":number<n_startup,"tpe_guided_trial":number>=n_startup,"suggested_parameters":params,"candidate_id":cid,"evaluation":result,"optimizer_objective":objective,"optuna_trial_state":str(study.trials[-1].state.name),"incumbent_q_before":before,"incumbent_q_after":incumbent_q,"incumbent_updated":bool(result.get("incumbent_updated")),"no_update_reason":result.get("no_update_reason","")}
            log.write(json.dumps(rec,ensure_ascii=False,allow_nan=True)+"\n"); records.append(rec)
    best=study.best_trial
    accepted_rows=[r for r in records if r["evaluation"]["execution_success"] and r["evaluation"]["accepted"]]
    best_accepted=max(accepted_rows,key=lambda r:r["evaluation"]["q"]) if accepted_rows else None
    trials_json=[{"number":t.number,"state":t.state.name,"value":t.value,"params":t.params} for t in study.trials]
    write_json(outdir/"optuna_study.json",{"direction":"maximize","trials":trials_json,"best_trial_number":best.number})
    optuna_best_id=candidate_hash(best.params)
    summary={"schema_version":"NON_PUBLICATION_SMOKE_TEST_v1","run_id":run_id,"method":"tpe","smoke_test":True,"publication_eligible":False,"requested_trials":5,"completed_trials":len(study.trials),"accepted_count":accepted,"failed_count":failed,"incumbent_update_count":updates,"baseline_score":baseline["q"],"baseline_accepted":baseline["accepted"],"final_incumbent_score":incumbent_q,"best_accepted_trial_candidate":None if best_accepted is None else best_accepted["candidate_id"],"benchmark_best_incumbent_candidate":incumbent_id,"best_optuna_trial_number":best.number,"best_optuna_trial_candidate":optuna_best_id,"best_optuna_trial_accepted":bool(next(r for r in records if r["trial_number"]==best.number)["evaluation"]["accepted"]),"optuna_best_matches_benchmark_incumbent":optuna_best_id==incumbent_id,"matlab_starts":6,"baseline_simulations":1,"candidate_simulations":5,"proposal_replay_simulations":0,"total_duration_s":time.perf_counter()-timer}
    write_json(outdir/"summary.json",summary)
    objectives=[r["optimizer_objective"] for r in records]; original=[{"trial":r["trial_number"],"candidate_id":r["candidate_id"]} for r in records]; replay=replay_proposals(space,objectives,seed,n_startup); mismatches=[i for i,(a,b) in enumerate(zip(original,replay)) if a["candidate_id"]!=b["candidate_id"]]
    write_json(outdir/"reproducibility_hashes.json",{"simulations_rerun":False,"exact_match":not mismatches,"mismatched_trial_indices":mismatches,"original":original,"replay":replay})
    return summary

def self_check():
    space=load_space(); assert len(space["variables"])==7; study=build_study(space); trial=study.ask(); p=suggest_parameters(trial,space); assert set(p)=={x["name"] for x in space["variables"]}; assert isinstance(p["horizon_max"],int); assert study.direction.name=="MAXIMIZE"; assert study.sampler._n_startup_trials==2; assert optimizer_objective({"execution_success":True,"accepted":True,"q":0.4})==0.4; assert optimizer_objective({"execution_success":False,"accepted":False,"q":0.4})==-0.6; assert "llm_wrapper" not in sys.modules; print("PASS: Python TPE self-checks")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--seed",type=int,default=SEED); ap.add_argument("--trials",type=int,default=5); ap.add_argument("--n-startup-trials",dest="n_startup_trials",type=int,default=STARTUP); args=ap.parse_args(); self_check() if args.self_test else print(json.dumps(run(args.seed,args.trials,args.n_startup_trials),indent=2))
