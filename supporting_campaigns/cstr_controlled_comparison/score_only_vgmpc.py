#!/usr/bin/env python3
"""Three-generation real-API score-only VG-MPC NON_PUBLICATION_SMOKE_TEST."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, socket, subprocess, sys, tempfile, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

from deepseek_requests_transport import request_chat_completion

ROOT=Path(__file__).resolve().parent; SPACE_PATH=ROOT/'search_space_cstr_reduced.json'; MATLAB=Path('/Applications/MATLAB_R2026a.app/bin/matlab')
METHOD='score_only_vgmpc'; SEED=20260807; MODEL='deepseek-chat'; TEMPERATURE=.3; MAX_TOKENS=512; TIMEOUT=60
SECURE_SEED=20260813; CONNECT_TIMEOUT=15; READ_TIMEOUT=60
PROMPT_VERSION='score_only_v1'; PARSER_VERSION='strict_json_v1'; ENDPOINT='https://api.deepseek.com/v1/chat/completions'
NAMES=['horizon_max','Wy_base','Wy_gain','Wu','Wdu_base','Wdu_gain','hessian_reg']
FORBIDDEN=['integral error','rms','tracking error','control energy','controller time','condition number','kkt','residual','violation','solver','metric vector','baseline metrics','detailed exception']
REQUIRED_MATLAB={'execution_success','metrics','s','v','q','accepted','incumbent_before','incumbent_after','incumbent_updated'}

def load_space():
    space=json.loads(SPACE_PATH.read_text(encoding='utf-8')); assert [v['name'] for v in space['variables']]==NAMES; return space

def get_api_key():
    key=os.environ.get('DEEPSEEK_API_KEY','')
    if not key: raise RuntimeError('DEEPSEEK_API_KEY is not set')
    return key

def ensure_output_safe(path):
    if path.exists() and any(path.iterdir()): raise FileExistsError(f'Refusing to overwrite non-empty output: {path}')

def build_prompt(space,incumbent,q,previous=None):
    lines=['Task: Propose values for exactly seven CSTR configuration parameters.','','Allowed parameters:']
    for s in space['variables']:
        typ='integer' if s['type']=='integer' else 'number'; lines.append(f"- {s['name']}: {typ}, range [{s['lo']}, {s['hi']}]")
    lines += ['','All other CSTR configuration settings are fixed to their baseline values.','Do not modify structural settings.','Objective: improve the accepted incumbent selection score.','','Current incumbent parameters:',json.dumps({n:incumbent[n] for n in NAMES},separators=(',',':')),'',f'Current incumbent q: {q:.17g}','','Previous candidate outcome:']
    if previous is None: lines.append('- previous_candidate_available: false')
    else:
        lines += ['- previous_candidate_available: true',f"- previous_candidate_q: {previous['q']:.17g}",f"- previous_candidate_accepted: {str(bool(previous['accepted'])).lower()}",f"- previous_candidate_execution_success: {str(bool(previous['execution_success'])).lower()}",f"- previous_candidate_incumbent_updated: {str(bool(previous['incumbent_updated'])).lower()}"]
    lines += ['','Output exactly one JSON object with exactly the seven parameter fields.','Use JSON numbers only. No explanation, no markdown, no extra fields.']
    return '\n'.join(lines)

def leakage_check(prompt):
    low=prompt.lower(); hits=[x for x in FORBIDDEN if x in low]
    return {'passed':not hits,'forbidden_hits':hits,'has_all_parameters':all(n in prompt for n in NAMES),'json_only_instruction':'No explanation, no markdown, no extra fields.' in prompt}

def strict_parse(raw,space):
    if '```' in raw: return None,'markdown_code_fence'
    try: data=json.loads(raw,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except Exception: return None,'malformed_json'
    if not isinstance(data,dict): return None,'not_object'
    if set(data)!=set(NAMES): return None,'field_set_mismatch'
    specs={x['name']:x for x in space['variables']}; out={}
    for n in NAMES:
        v=data[n]
        if isinstance(v,bool) or not isinstance(v,(int,float)): return None,'invalid_numeric_type'
        if not math.isfinite(float(v)): return None,'non_finite'
        if n=='horizon_max' and (not isinstance(v,int) or isinstance(v,bool)): return None,'non_integer_horizon'
        if v<specs[n]['lo'] or v>specs[n]['hi']: return None,'out_of_range'
        out[n]=v
    canonical=json.dumps(data,separators=(',',':'),allow_nan=False)
    if json.loads(canonical)!=data: return None,'non_canonical_json'
    return out,''

def hash_text(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()
def secure_write(path,text,secret=''):
    if secret and secret in text: raise RuntimeError('credential_redaction_failure')
    path.write_text(text,encoding='utf-8'); os.chmod(path,0o600)

def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    return value

def credential_scan(path,secret):
    failures=[]
    for artifact in sorted(p for p in path.rglob('*') if p.is_file()):
        data=artifact.read_bytes()
        if secret.encode('utf-8') in data:
            failures.append(str(artifact.relative_to(path)))
    return {'credential_scan_passed':not failures,'credential_matches':failures}

def transport_category(exc):
    if isinstance(exc,urllib.error.HTTPError):
        if exc.code==429:return 'http_429'
        if 500<=exc.code<=599:return 'http_5xx'
        return 'http_non_retryable'
    if isinstance(exc,(TimeoutError,socket.timeout)):return 'timeout'
    if isinstance(exc,urllib.error.URLError):
        reason=exc.reason
        if isinstance(reason,(TimeoutError,socket.timeout)):return 'timeout'
        if isinstance(reason,(ConnectionResetError,ConnectionError)):return 'connection_reset'
        return 'transport_non_retryable'
    if isinstance(exc,(ConnectionResetError,ConnectionError)):return 'connection_reset'
    return 'transport_non_retryable'

def redact_sensitive(text, secret=''):
    value=str(text or '')
    if secret:
        value=value.replace(secret,'[REDACTED]')
    value=re.sub(r'(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;"}\]]+', r'\1[REDACTED]', value)
    value=re.sub(r'(?i)\bbearer\s+[-A-Za-z0-9._~+/]+=*', 'Bearer [REDACTED]', value)
    value=re.sub(r'(?i)\bsk-[A-Za-z0-9_-]{6,}', '[REDACTED]', value)
    value=re.sub(r'\*{2,}[A-Za-z0-9_-]{2,}', '[REDACTED]', value)
    value=re.sub(r'(?i)(["\']?(?:api[_-]?key|apikey|access[_-]?token|token)["\']?\s*[:=]\s*["\']?)[^"\'\s,&}]+', r'\1[REDACTED]', value)
    value=re.sub(r'(?i)([?&](?:api[_-]?key|apikey|access[_-]?token|token)=)[^&#\s]+', r'\1[REDACTED]', value)
    return value

def classify_http_status(status):
    if status == 429:
        return True, 'http_429'
    if 500 <= status <= 599:
        return True, 'http_5xx'
    return False, 'http_non_retryable'

def safe_http_error(exc, duration_s, secret=''):
    status=getattr(exc,'code',None); headers=getattr(exc,'headers',None)
    reason=redact_sensitive(getattr(exc,'reason','') or '',secret)
    body=''
    try:
        body=exc.read(8192).decode('utf-8','replace')
    except Exception:
        body=''
    redacted=redact_sensitive(body,secret)[:2000]
    content_type=''
    if headers is not None:
        content_type=redact_sensitive(headers.get('Content-Type',''),secret)
    provider_type=provider_code=''
    provider_message=redacted
    try:
        parsed=json.loads(body)
        err=parsed.get('error',parsed) if isinstance(parsed,dict) else {}
        if isinstance(err,dict):
            provider_type=redact_sensitive(err.get('type',''),secret)
            provider_code=redact_sensitive(err.get('code',''),secret)
            provider_message=redact_sensitive(err.get('message',redacted),secret)[:2000]
    except Exception:
        pass
    retryable, classification=classify_http_status(status) if isinstance(status,int) else (False,'transport_non_retryable')
    parts=urllib.parse.urlsplit(ENDPOINT)
    return {'http_status':status,'http_reason':reason,'endpoint_host':parts.hostname or '',
            'endpoint_path':parts.path,'request_method':'POST','provider':'deepseek',
            'model':MODEL,'response_content_type':content_type,'provider_error_type':provider_type,
            'provider_error_code':provider_code,'provider_message_redacted':provider_message,
            'retryable':retryable,'retry_classification_reason':classification,
            'request_duration_s':duration_s}

def call_transport_once(prompt,key):
    payload={'model':MODEL,'messages':[{'role':'system','content':'Return only the requested JSON object.'},{'role':'user','content':prompt}],'temperature':TEMPERATURE,'max_tokens':MAX_TOKENS,'response_format':{'type':'json_object'}}
    body=json.dumps(payload).encode(); started=time.perf_counter(); req=urllib.request.Request(ENDPOINT,data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(req,timeout=TIMEOUT) as response:
            result=json.loads(response.read().decode('utf-8')); return {'success':True,'content':result['choices'][0]['message']['content'],'usage':result.get('usage',{}),'physical_requests':1,'error':None,'request_duration_s':time.perf_counter()-started}
    except urllib.error.HTTPError as exc:
        return {'success':False,'content':'','usage':{},'physical_requests':1,'error':safe_http_error(exc,time.perf_counter()-started,key),'request_duration_s':time.perf_counter()-started}
    except Exception as exc:
        category=transport_category(exc); parts=urllib.parse.urlsplit(ENDPOINT)
        return {'success':False,'content':'','usage':{},'physical_requests':1,'error':{'http_status':None,'http_reason':'','endpoint_host':parts.hostname or '','endpoint_path':parts.path,'request_method':'POST','provider':'deepseek','model':MODEL,'response_content_type':'','provider_error_type':'','provider_error_code':'','provider_message_redacted':'','retryable':False,'retry_classification_reason':category,'request_duration_s':time.perf_counter()-started},'request_duration_s':time.perf_counter()-started}

def call_api(prompt,key):
    payload={'model':MODEL,'messages':[{'role':'system','content':'Return only the requested JSON object.'},{'role':'user','content':prompt}],'temperature':TEMPERATURE,'max_tokens':MAX_TOKENS,'response_format':{'type':'json_object'}}
    body=json.dumps(payload).encode(); physical=0; retry_reason=''; started=time.perf_counter()
    for attempt in range(2):
        physical+=1; req=urllib.request.Request(ENDPOINT,data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(req,timeout=TIMEOUT) as response: result=json.loads(response.read().decode('utf-8'))
            content=result['choices'][0]['message']['content']; return content,result.get('usage',{}),physical,retry_reason,time.perf_counter()-started,''
        except Exception as exc:
            category=transport_category(exc); retryable=category in {'timeout','connection_reset','http_429','http_5xx'}
            if attempt==0 and retryable: retry_reason=category; continue
            return '',{},physical,retry_reason,time.perf_counter()-started,category

def matlab_call(request):
    with tempfile.TemporaryDirectory(prefix='score_only_bridge_') as td:
        inp=Path(td)/'input.json'; out=Path(td)/'output.json'; inp.write_text(json.dumps(request),encoding='utf-8')
        esc=lambda p:str(p).replace("'","''"); expr=f"addpath('{esc(ROOT)}'); evaluate_score_only_candidate('{esc(inp)}','{esc(out)}')"
        proc=subprocess.run([str(MATLAB),'-batch',expr],text=True,capture_output=True)
        if proc.returncode!=0 or not out.exists(): raise RuntimeError('matlab_bridge_failure')
        result=json.loads(out.read_text(encoding='utf-8')); missing=sorted(REQUIRED_MATLAB-set(result))
        if missing: raise RuntimeError('matlab_result_missing_fields')
        return result

def mock_generation_guard(responses,space,baseline):
    simulations=0; logical=0
    for raw in responses[:3]:
        logical+=1; valid,_=strict_parse(raw,space); simulations+=int(valid is not None)
    return logical,simulations

def run_secure(api_key, output_dir, transport_fn=request_chat_completion, matlab_fn=matlab_call,
               seed=SECURE_SEED):
    """Run exactly three Score-only logical generations using an in-memory key."""
    if not isinstance(api_key,str) or not api_key:
        raise ValueError('api_key must be a non-empty string')
    if not isinstance(output_dir,Path): output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite non-empty output: {output_dir}')
    space=load_space(); output_dir.mkdir(parents=True,exist_ok=True); os.chmod(output_dir,0o700)
    prompts=output_dir/'prompts'; responses=output_dir/'responses'
    prompts.mkdir(); responses.mkdir(); os.chmod(prompts,0o700); os.chmod(responses,0o700)
    write_json=lambda path,data: secure_write(path,json.dumps(json_safe(data),indent=2,ensure_ascii=False,allow_nan=False)+'\n',api_key)
    manifest={'schema_version':'NON_PUBLICATION_SMOKE_TEST_v1','method':METHOD,'smoke_test':True,
        'publication_eligible':False,'provider':'deepseek','endpoint':ENDPOINT,
        'backend':'requests.post','model':MODEL,'temperature':TEMPERATURE,'max_tokens':MAX_TOKENS,
        'connect_timeout_s':CONNECT_TIMEOUT,'read_timeout_s':READ_TIMEOUT,
        'response_format':{'type':'json_object'},'prompt_version':PROMPT_VERSION,
        'parser_version':PARSER_VERSION,'run_seed':seed,'logical_generations':3,
        'fallback_generations':0,'fourth_generation_prohibited':True,
        'api_key_entered_securely':True,'api_key_saved':False,
        'authorization_header_saved':False,'local_credential_read':False,
        'credential_loaded_from_environment':False}
    write_json(output_dir/'run_manifest.json',manifest)
    baseline=matlab_fn({'mode':'baseline'})
    write_json(output_dir/'baseline.json',baseline)
    incumbent=baseline['candidate']; incumbent_q=float(baseline['q'])
    incumbent_id=baseline.get('candidate_id','baseline'); previous=None
    physical_total=retry_total=valid_count=invalid_count=simulations=accepted=failed=updates=0
    retry_reasons=[]
    rows=[]; leakage_rows=[]; valid_details=[]; started=time.perf_counter()
    log_path=output_dir/'evaluations.jsonl'; secure_write(log_path,'',api_key)
    for generation in range(1,4):
        prompt=build_prompt(space,incumbent,incumbent_q,previous)
        leak=leakage_check(prompt)
        if not leak['passed'] or not leak['has_all_parameters'] or not leak['json_only_instruction']:
            raise RuntimeError('prompt_leakage_test_failed')
        prompt_path=prompts/f'generation_{generation:03d}.txt'; secure_write(prompt_path,prompt,api_key)
        saved_prompt=prompt_path.read_text(encoding='utf-8'); saved_leak=leakage_check(saved_prompt)
        leakage_rows.append({'generation':generation,'before_request':leak,'saved_prompt_check':saved_leak})
        if not saved_leak['passed'] or api_key in saved_prompt:
            raise RuntimeError('prompt_artifact_security_check_failed')
        request_result=transport_fn(api_key=api_key,
            messages=[{'role':'system','content':'Return only the requested JSON object.'},
                      {'role':'user','content':prompt}],model=MODEL,temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,connect_timeout=CONNECT_TIMEOUT,read_timeout=READ_TIMEOUT,
            response_format={'type':'json_object'},allow_transport_retry=True)
        physical=int(request_result.get('physical_request_count',0)); physical_total+=physical
        retry_total+=int(request_result.get('retry_count',max(0,physical-1)))
        retry_reasons.extend(request_result.get('retry_reasons',[]))
        if physical < 1 or physical > 2 or physical_total > 6:
            raise RuntimeError('generation_request_count_limit_exceeded')
        raw=request_result.get('assistant_content') or ''
        credential_echo=api_key in raw
        if credential_echo: raw=''; transport_error='credential_echo_detected'
        else: transport_error=request_result.get('transport_exception_class') or request_result.get('provider_message_redacted') or ''
        response_path=responses/f'generation_{generation:03d}.txt'
        response_body=raw if raw else json.dumps({'transport_error_category':transport_error or 'empty_response'},separators=(',',':'))
        secure_write(response_path,response_body,api_key)
        response_text=response_path.read_text(encoding='utf-8'); response_hash=hash_text(response_text)
        parse_success=False
        if raw:
            try: json.loads(raw); parse_success=True
            except Exception: parse_success=False
        params,validation_error=(strict_parse(raw,space) if raw else (None,transport_error or 'empty_response'))
        schema_success=params is not None
        before=incumbent_q
        evaluation={'execution_success':False,'metrics':[],'s':float('-inf'),'v':100.0,
            'q':float('-inf'),'accepted':False,'incumbent_before':before,
            'incumbent_after':before,'incumbent_updated':False,
            'simulation_duration_s':0.0,'no_update_reason':'invalid_api_output'}
        candidate_id=''; candidate_hash=''
        if params is not None:
            valid_count+=1; simulations+=1
            evaluation=matlab_fn({'mode':'candidate','parameters':params,
                'incumbent_before':before,'incumbent_candidate_id':incumbent_id})
            candidate_id=evaluation.get('candidate_id',''); candidate_hash=candidate_id
            if evaluation.get('incumbent_updated'):
                incumbent=evaluation['candidate']; incumbent_q=float(evaluation['incumbent_after']); incumbent_id=candidate_id; updates+=1
            accepted+=int(bool(evaluation.get('accepted'))); failed+=int(not evaluation.get('execution_success'))
            valid_details.append({'hash':candidate_hash,'parameters':params,'q':evaluation.get('q'),
                'accepted':bool(evaluation.get('accepted')),'incumbent_updated':bool(evaluation.get('incumbent_updated'))})
        else:
            invalid_count+=1; failed+=1
        previous={k:evaluation[k] for k in ('q','accepted','execution_success','incumbent_updated')}
        rec={'schema_version':'NON_PUBLICATION_SMOKE_TEST_v1','method':METHOD,'smoke_test':True,
            'publication_eligible':False,'logical_generation':generation,
            'physical_request_count':physical,'retry_count':request_result.get('retry_count',max(0,physical-1)),
            'retry_reason':request_result.get('retry_reasons',[]),'prompt_hash':hash_text(saved_prompt),
            'response_hash':response_hash,'parse_success':parse_success,
            'schema_validation_success':schema_success,'validation_error_category':validation_error,
            'candidate_hash':candidate_hash,'parameters':params,
            'execution_success':bool(evaluation.get('execution_success')),'internal_metrics':evaluation.get('metrics',[]),
            's':evaluation.get('s'),'v':evaluation.get('v'),'q':evaluation.get('q'),
            'accepted':bool(evaluation.get('accepted')),'incumbent_before':evaluation.get('incumbent_before'),
            'incumbent_after':evaluation.get('incumbent_after'),'incumbent_updated':bool(evaluation.get('incumbent_updated')),
            'reason_for_no_update':evaluation.get('no_update_reason',''),'api_duration_s':request_result.get('request_duration_s',0.0),
            'simulation_duration_s':evaluation.get('simulation_duration_s',0.0),'token_usage':request_result.get('token_usage',{}),
            'fallback_used':False,'provider_http_status':request_result.get('http_status'),
            'provider_error_type':request_result.get('provider_error_type',''),
            'provider_error_code':request_result.get('provider_error_code',''),
            'provider_message_redacted':request_result.get('provider_message_redacted',''),
            'transport_exception_class':request_result.get('transport_exception_class',''),
            'credential_echo_detected':credential_echo}
        line=json.dumps(json_safe(rec),ensure_ascii=False,allow_nan=False)+'\n'
        if api_key in line: raise RuntimeError('credential_redaction_failure')
        with log_path.open('a',encoding='utf-8') as log: log.write(line)
        rows.append(rec)
    summary={'schema_version':'NON_PUBLICATION_SMOKE_TEST_v1','method':METHOD,'smoke_test':True,
        'publication_eligible':False,'baseline_q':baseline.get('q'),'baseline_accepted':baseline.get('accepted'),
        'logical_generations':3,'physical_generation_requests':physical_total,
        'generation_retry_count':retry_total,'generation_retry_reasons':retry_reasons,
        'valid_candidates':valid_count,'invalid_responses':invalid_count,'candidate_simulations':simulations,
        'accepted_count':accepted,'failed_count':failed,'incumbent_updates':updates,
        'final_incumbent_score':incumbent_q,'final_incumbent_candidate_id':incumbent_id,
        'baseline_simulations':1,'fallback_generations':0,'fourth_generation_started':False,
        'valid_candidate_details':valid_details,'total_duration_s':time.perf_counter()-started}
    write_json(output_dir/'summary.json',summary)
    scan=credential_scan(output_dir,api_key)
    validation={'prompt_leakage_checks':leakage_rows,'credential_scan_passed':scan['credential_scan_passed'],
        'credential_matches':scan['credential_matches'],'api_key_entered_securely':True,
        'api_key_saved':False,'authorization_header_saved':False,'local_credential_read':False,
        'physical_request_count':physical_total,'logical_generation_count':len(rows),
        'baseline_simulation_count':1,'candidate_simulation_count':simulations,
        'fallback_used':False,'fourth_generation_started':False,
        'full_feedback_started':False,
        'candidate_simulations_not_above_valid_candidates':simulations<=valid_count}
    write_json(output_dir/'validation_report.json',validation)
    final_scan=credential_scan(output_dir,api_key)
    if not final_scan['credential_scan_passed']: raise RuntimeError('credential_scan_failed')
    return summary

def run(seed=SEED,generations=3):
    if generations!=3: raise ValueError('Exactly three logical generations are required')
    key=get_api_key(); outdir=ROOT/'NON_PUBLICATION_SMOKE_TEST'/f'score_only_vgmpc_seed_{seed}'; ensure_output_safe(outdir)
    space=load_space(); outdir.mkdir(parents=True,exist_ok=True); os.chmod(outdir,0o700); prompts=outdir/'prompts'; responses=outdir/'responses'; prompts.mkdir(); responses.mkdir(); os.chmod(prompts,0o700); os.chmod(responses,0o700)
    manifest={'schema_version':'NON_PUBLICATION_SMOKE_TEST_v1','method':METHOD,'smoke_test':True,'publication_eligible':False,'provider':'deepseek','backend':'openai_compatible_chat_completions','model':MODEL,'temperature':TEMPERATURE,'max_tokens':MAX_TOKENS,'timeout_s':TIMEOUT,'request_format':'json_object','prompt_version':PROMPT_VERSION,'parser_version':PARSER_VERSION,'credential_loaded_from_environment':True,'local_credential_read_attempted':False,'run_seed':seed,'api_output_deterministic_from_seed':False}
    write=lambda p,d: secure_write(p,json.dumps(d,indent=2,ensure_ascii=False,allow_nan=True)); write(outdir/'run_manifest.json',manifest)
    baseline=matlab_call({'mode':'baseline'}); write(outdir/'baseline.json',baseline); incumbent=baseline['candidate']; incumbent_q=float(baseline['q']); incumbent_id=baseline['candidate_id']; previous=None
    physical_total=valid_count=invalid_count=simulations=accepted=failed=updates=0; rows=[]; leakage_rows=[]; timer=time.perf_counter()
    with (outdir/'evaluations.jsonl').open('w',encoding='utf-8') as log:
        os.chmod(outdir/'evaluations.jsonl',0o600)
        for generation in range(1,4):
            prompt=build_prompt(space,incumbent,incumbent_q,previous); leak=leakage_check(prompt)
            if not leak['passed'] or not leak['has_all_parameters'] or not leak['json_only_instruction']: raise RuntimeError('prompt_leakage_test_failed')
            if key in prompt: raise RuntimeError('prompt_artifact_security_check_failed')
            prompt_path=prompts/f'generation_{generation:03d}.txt'; secure_write(prompt_path,prompt); prompt_hash=hash_text(prompt_path.read_text())
            leak_after_save=leakage_check(prompt_path.read_text()); leakage_rows.append({'generation':generation,'before_request':leak,'saved_prompt_check':leak_after_save})
            if not leak_after_save['passed'] or key in prompt_path.read_text(): raise RuntimeError('prompt_artifact_security_check_failed')
            raw,usage,physical,retry_reason,api_duration,transport_error=call_api(prompt,key); physical_total+=physical
            credential_echo=bool(raw and key in raw)
            if credential_echo:
                raw=''; transport_error='credential_echo_detected'
            response_path=responses/f'generation_{generation:03d}.txt'; secure_write(response_path,raw if raw else json.dumps({'transport_error_category':transport_error})); response_text=response_path.read_text(); response_hash=hash_text(response_text)
            credential_absent=key not in response_text; params,validation_error=(strict_parse(raw,space) if raw else (None,transport_error or 'empty_response')); parse_success=bool(raw and validation_error not in {'malformed_json','markdown_code_fence'}); schema_success=params is not None
            before=incumbent_q; evaluation={'execution_success':False,'metrics':[],'s':float('-inf'),'v':100,'q':float('-inf'),'accepted':False,'incumbent_before':before,'incumbent_after':before,'incumbent_updated':False,'simulation_duration_s':0,'no_update_reason':'invalid_api_output'}; candidate_id=''
            if params is not None:
                valid_count+=1; simulations+=1; evaluation=matlab_call({'mode':'candidate','parameters':params,'incumbent_before':before,'incumbent_candidate_id':incumbent_id}); candidate_id=evaluation.get('candidate_id','')
                if evaluation['incumbent_updated']:
                    incumbent=evaluation['candidate']; incumbent_q=float(evaluation['incumbent_after']); incumbent_id=candidate_id; updates+=1
                accepted+=int(bool(evaluation['accepted'])); failed+=int(not evaluation['execution_success'])
            else: invalid_count+=1; failed+=1
            previous={k:evaluation[k] for k in ('q','accepted','execution_success','incumbent_updated')}
            rec={'schema_version':'NON_PUBLICATION_SMOKE_TEST_v1','method':METHOD,'smoke_test':True,'publication_eligible':False,'logical_generation':generation,'physical_request_count':physical,'retry_count':physical-1,'retry_reason':retry_reason,'backend':'deepseek_openai_compatible','model':MODEL,'temperature':TEMPERATURE,'prompt_version':PROMPT_VERSION,'parser_version':PARSER_VERSION,'prompt_hash':prompt_hash,'response_hash':response_hash,'parse_success':parse_success,'schema_validation_success':schema_success,'validation_error_category':validation_error,'parameters':params,'candidate_id':candidate_id,'evaluation':evaluation,'api_duration_s':api_duration,'token_usage':usage,'fallback_used':False,'prompt_leakage_passed':leak_after_save['passed'],'credential_absent_from_artifacts':credential_absent,'credential_echo_detected':credential_echo}
            log.write(json.dumps(rec,ensure_ascii=False,allow_nan=True)+'\n'); log.flush(); rows.append(rec)
    summary={'schema_version':'NON_PUBLICATION_SMOKE_TEST_v1','method':METHOD,'smoke_test':True,'publication_eligible':False,'baseline_q':baseline['q'],'baseline_accepted':baseline['accepted'],'logical_generations':3,'physical_api_requests':physical_total,'valid_candidates':valid_count,'invalid_responses':invalid_count,'candidate_simulations':simulations,'accepted_count':accepted,'failed_count':failed,'incumbent_updates':updates,'final_incumbent_score':incumbent_q,'final_incumbent_candidate_id':incumbent_id,'baseline_simulations':1,'fallback_generations':0,'fourth_generation_started':False,'total_duration_s':time.perf_counter()-timer}
    artifact_paths=list(prompts.glob('*'))+list(responses.glob('*'))
    validation={'prompt_leakage_checks':leakage_rows,'credential_loaded_from_environment':True,'local_credential_read_attempted':False,'credential_scan_passed':all(key not in p.read_text(encoding='utf-8') for p in artifact_paths),'logical_generation_guard_passed':len(rows)==3,'candidate_simulations_not_above_valid_candidates':simulations<=valid_count,'fallback_used':False,'fourth_generation_started':False}
    write(outdir/'summary.json',summary); write(outdir/'validation_report.json',validation); return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=SEED); ap.add_argument('--generations',type=int,default=3); ap.add_argument('--mode',default='score_only'); args=ap.parse_args()
    if args.mode!='score_only': raise ValueError('Only score_only mode is supported')
    print(json.dumps(run(args.seed,args.generations),indent=2))
