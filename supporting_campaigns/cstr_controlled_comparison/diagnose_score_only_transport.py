#!/usr/bin/env python3
"""One-request transport diagnosis; never runs MATLAB or scientific evaluation."""
from __future__ import annotations
import hashlib,json,os,sys,time
from pathlib import Path
import score_only_vgmpc as transport

ROOT=Path(__file__).resolve().parent; SEED=20260808

def secure(path,data,secret):
    text=json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)
    if secret in text: raise RuntimeError('credential_redaction_failure')
    path.write_text(text,encoding='utf-8'); os.chmod(path,0o600)

def tree_fingerprint(path):
    h=hashlib.sha256()
    for f in sorted(x for x in path.rglob('*') if x.is_file()):
        h.update(str(f.relative_to(path)).encode()); h.update(b'\0'); h.update(f.read_bytes()); h.update(b'\0')
    return h.hexdigest()

def env_checks(key):
    return {'credential_present':bool(key),'credential_has_surrounding_quotes':bool(key and len(key)>=2 and key[0]==key[-1] and key[0] in "\"'"),'credential_has_outer_whitespace':bool(key and key!=key.strip()),'endpoint_nonempty':bool(transport.ENDPOINT),'model_nonempty':bool(transport.MODEL),'ssl_verification_enabled':True}

def classify_root(error):
    if not error:return 'success'
    status=error.get('http_status'); msg=(error.get('provider_message_redacted') or '').lower()
    if status==401:return 'authentication failure'
    if status==403:return 'authorization or account failure'
    if status in (400,422):
        if 'response_format' in msg or 'json_object' in msg:return 'response_format incompatibility'
        return 'malformed request body'
    if status==404:return 'endpoint/path or model-not-found failure'
    if status==429:return 'rate limit'
    if isinstance(status,int) and 500<=status<=599:return 'provider server error'
    return 'network/TLS/DNS failure' if status is None else 'unknown'

def run(root=ROOT, request_fn=transport.call_transport_once):
    key=transport.get_api_key(); prior=root/'NON_PUBLICATION_SMOKE_TEST'/'score_only_vgmpc_seed_20260807'; prior_fp=tree_fingerprint(prior)
    out=root/'NON_PUBLICATION_SMOKE_TEST'/'score_only_transport_diagnosis_20260808'; transport.ensure_output_safe(out)
    out.mkdir(parents=True,exist_ok=True); os.chmod(out,0o700)
    prompt='Return exactly:\n{"diagnostic": true}'
    result=request_fn(prompt,key)
    rec={'diagnostic_only':True,'publication_eligible':False,'scientific_candidate_evaluation':False,'matlab_simulation_count':0,'physical_api_requests':result['physical_requests'],'endpoint':transport.ENDPOINT,'model':transport.MODEL,'request_body_fields':['model','messages','temperature','max_tokens','response_format'],'request_method':'POST','response_format':{'type':'json_object'},'result_success':result['success'],'error':result['error'],'root_cause_category':classify_root(result['error']),'proposed_correction':'Determine from the recorded status and provider message; do not apply automatically.','prior_score_only_tree_unchanged':tree_fingerprint(prior)==prior_fp,'full_feedback_started':False}
    secure(out/'diagnosis.json',rec,key)
    return rec

if __name__=='__main__': print(json.dumps(run(),indent=2,ensure_ascii=False))
