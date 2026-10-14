#!/usr/bin/env python3
"""One-request DeepSeek authentication validation, NON_PUBLICATION_SMOKE_TEST."""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
ENDPOINT='https://api.deepseek.com/v1/chat/completions'; MODEL='deepseek-chat'
COMMON={'diagnostic_only':True,'publication_eligible':False,'scientific_candidate_evaluation':False,'controller_generation':False,'api_key_read_from_environment':True,'local_credential_read':False,'authorization_header_saved':False,'matlab_simulation_count':0,'physical_request_count':1,'retry_count':0}

def redact(value):
    text=str(value or '')
    text=re.sub(r'(?i)authorization\s*[:=]\s*(?:bearer\s+)?[^\s,;}]+','Authorization: [REDACTED]',text)
    text=re.sub(r'(?i)\bbearer\s+[-A-Za-z0-9._~+/]+=*','Bearer [REDACTED]',text)
    text=re.sub(r'(?i)\bsk-[A-Za-z0-9_-]{6,}','[REDACTED]',text)
    text=re.sub(r'(?i)(["\']?(?:api[_-]?key|apikey|access[_-]?token|token)["\']?\s*[:=]\s*["\']?)[^"\'\s,&}]+',r'\1[REDACTED]',text)
    text=re.sub(r'\*{2,}[A-Za-z0-9_-]{2,}','[REDACTED]',text)
    return text[:2000]

def proxy_summary():
    result={}
    for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy'):
        raw=os.environ.get(name)
        if not raw: result[name]={'set':False}; continue
        from urllib.parse import urlsplit
        try:
            p=urlsplit(raw if '://' in raw else 'http://'+raw); result[name]={'set':True,'scheme':p.scheme,'host':p.hostname or '','port':p.port}
        except Exception: result[name]={'set':True,'scheme':'','host':'','port':None}
    return result

def get_key():
    key=os.environ.get('DEEPSEEK_API_KEY','')
    if not key: raise RuntimeError('DEEPSEEK_API_KEY missing')
    return key

def classify_status(status):
    if status==401:return 'authentication failure',False
    if status==403:return 'authorization or account failure',False
    if status==429:return 'rate limit',False
    if isinstance(status,int) and 500<=status<=599:return 'provider server error',True
    return 'http failure',False

def error_from_response(response, elapsed):
    provider_type=provider_code=message=''
    try:
        data=response.json(); err=data.get('error',data) if isinstance(data,dict) else {}
        if isinstance(err,dict):
            provider_type=redact(err.get('type','')); provider_code=redact(err.get('code','')); message=redact(err.get('message',''))
    except Exception: message='invalid provider error JSON'
    category,retryable=classify_status(response.status_code)
    return {'http_status':response.status_code,'http_reason':redact(response.reason),'response_content_type':redact(response.headers.get('Content-Type','')),'provider_error_type':provider_type,'provider_error_code':provider_code,'provider_message_redacted':message,'retryable':retryable,'retry_classification_reason':'http_5xx' if retryable else 'http_non_retryable','redirect_history':[{'status':h.status_code,'host':redact(getattr(h.url,'host','')) if hasattr(h.url,'host') else ''} for h in response.history],'request_duration_s':elapsed}

def error_from_exception(exc, elapsed):
    reason=getattr(exc,'reason',None)
    return {'http_status':None,'http_reason':'','response_content_type':'','provider_error_type':'','provider_error_code':'','provider_message_redacted':redact(exc),'retryable':False,'retry_classification_reason':type(exc).__name__,'transport_exception_class':type(exc).__name__,'transport_exception_module':type(exc).__module__,'nested_reason_class':type(reason).__name__ if reason is not None else '','nested_reason_text':redact(reason),'errno':getattr(exc,'errno',None),'connect_timeout':isinstance(exc,requests.exceptions.ConnectTimeout),'read_timeout':isinstance(exc,requests.exceptions.ReadTimeout),'ssl_exception':isinstance(exc,requests.exceptions.SSLError),'proxy_exception':isinstance(exc,requests.exceptions.ProxyError),'redirect_history':[],'request_duration_s':elapsed}

def write_json(path,data,key):
    text=json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)
    if key in text: raise RuntimeError('credential redaction failure')
    path.write_text(text+'\n',encoding='utf-8'); os.chmod(path,0o600)

def run(root=ROOT, post=requests.post):
    out=root/'NON_PUBLICATION_SMOKE_TEST'/'score_only_auth_requests_validation_20260811'
    if out.exists() and any(out.iterdir()): raise FileExistsError('Refusing to overwrite non-empty output')
    key=get_key()
    out.mkdir(parents=True,exist_ok=True); os.chmod(out,0o700)
    payload={'model':MODEL,'messages':[{'role':'system','content':'Return only the requested JSON object.'},{'role':'user','content':'Return exactly:\n{"authentication_valid": true}'}],'temperature':0.3,'max_tokens':64,'response_format':{'type':'json_object'}}
    started=time.perf_counter(); record={**COMMON,'python_executable':sys.executable,'python_version':sys.version,'requests_version':requests.__version__,'endpoint_host':'api.deepseek.com','endpoint_path':'/v1/chat/completions','model':MODEL,'request_body_fields':list(payload),'proxy_summary':proxy_summary(),'api_key_present':True,'api_key_nonempty':True}
    try:
        response=post(ENDPOINT,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},json=payload,timeout=(15,60),verify=True,allow_redirects=True); elapsed=time.perf_counter()-started
        record.update({'http_response_received':True,'http_status':response.status_code,'http_reason':redact(response.reason),'response_content_type':redact(response.headers.get('Content-Type','')),'redirect_history':[{'status':h.status_code} for h in response.history],'request_duration_s':elapsed})
        try:
            envelope=response.json(); content=envelope.get('choices',[{}])[0].get('message',{}).get('content',''); parsed=json.loads(content); valid=True
        except Exception:
            content=''; parsed=None; valid=False
        record.update({'provider_response_json_valid':valid,'assistant_content_present':bool(content),'parsed_diagnostic_response':parsed,'authentication_valid':bool(response.ok and valid and parsed.get('authentication_valid') is True) if valid and isinstance(parsed,dict) else False,'provider_error_type':'','provider_error_code':'','provider_message_redacted':'' if response.ok else error_from_response(response,elapsed)['provider_message_redacted']})
    except Exception as exc:
        record.update({'http_response_received':False,'http_status':None,'http_reason':'','response_content_type':'','redirect_history':[],'request_duration_s':time.perf_counter()-started,'provider_response_json_valid':False,'assistant_content_present':False,'parsed_diagnostic_response':None,'authentication_valid':False,'provider_error_type':'','provider_error_code':'','provider_message_redacted':'','transport_error':error_from_exception(exc,time.perf_counter()-started)})
    write_json(out/'auth_requests_validation.json',record,key)
    write_json(out/'environment_summary.json',{**COMMON,'api_key_present':True,'api_key_nonempty':True,'proxy_summary':proxy_summary(),'python_executable':sys.executable,'python_version':sys.version,'requests_version':requests.__version__},key)
    write_json(out/'validation_report.json',{**COMMON,'request_succeeded':bool(record.get('authentication_valid')),'physical_request_count':1,'retry_count':0,'matlab_simulation_count':0,'score_only_rerun':False,'full_feedback_started':False,'credential_scan_passed':True},key)
    return record

if __name__=='__main__': print(json.dumps(run(),indent=2,ensure_ascii=False))
