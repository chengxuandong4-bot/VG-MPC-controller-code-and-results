#!/usr/bin/env python3
"""NON_PUBLICATION_SMOKE_TEST local DNS/TCP/TLS/HTTPS diagnosis."""
from __future__ import annotations
import json, os, platform, re, socket, ssl, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

HOST='api.deepseek.com'; PORT=443; HEAD_URL='https://api.deepseek.com/'
ROOT=Path(__file__).resolve().parent
ENV_PROXY=('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy')
ENV_CERT=('SSL_CERT_FILE','SSL_CERT_DIR','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE')
COMMON={'diagnostic_only':True,'publication_eligible':False,'scientific_candidate_evaluation':False,'api_key_read':False,'authorization_header_created':False,'matlab_simulation_count':0}

def redact(text):
    value=str(text or '')
    value=re.sub(r'(?i)authorization\s*[:=]\s*(?:bearer\s+)?[^\s,;}]+','Authorization: [REDACTED]',value)
    value=re.sub(r'(?i)\bbearer\s+[-A-Za-z0-9._~+/]+=*','Bearer [REDACTED]',value)
    value=re.sub(r'(?i)\bsk-[A-Za-z0-9_-]{6,}','[REDACTED]',value)
    value=re.sub(r'(?i)(["\']?(?:api[_-]?key|apikey|access[_-]?token|token)["\']?\s*[:=]\s*["\']?)[^"\'\s,&}]+',r'\1[REDACTED]',value)
    value=re.sub(r'(?i)([?&](?:api[_-]?key|apikey|access[_-]?token|token)=)[^&#\s]+',r'\1[REDACTED]',value)
    value=re.sub(r'\*{2,}[A-Za-z0-9_-]{2,}','[REDACTED]',value)
    return value[:2000]

def safe_text(value):
    try: return str(value or '')
    except Exception: return ''

def safe_attr(obj,name,default=None):
    try: return getattr(obj,name,default)
    except Exception: return default

def exception_details(exc, elapsed, hostname=HOST):
    reason=getattr(exc,'reason',None)
    details={'exception_class':type(exc).__name__,'exception_module':type(exc).__module__,'exception_text':redact(safe_text(exc)),'errno':safe_attr(exc,'errno'),'strerror':redact(safe_text(safe_attr(exc,'strerror',''))),'nested_reason_class':type(reason).__name__ if reason is not None else '','nested_reason_module':type(reason).__module__ if reason is not None else '','nested_reason_text':redact(safe_text(reason)),'dns_error_code':safe_attr(exc,'errno') if isinstance(exc,socket.gaierror) else None,'ssl_verify_code':safe_attr(exc,'verify_code'),'ssl_verify_message':redact(safe_text(safe_attr(exc,'verify_message',''))),'socket_error_code':safe_attr(exc,'errno') if isinstance(exc,OSError) else None,'timeout':isinstance(exc,(TimeoutError,socket.timeout)) or isinstance(reason,(TimeoutError,socket.timeout)),'hostname':hostname,'elapsed_s':elapsed}
    return details

def proxy_summary():
    result={}
    for name in ENV_PROXY:
        raw=os.environ.get(name)
        if not raw: result[name]={'set':False}; continue
        try:
            p=urllib.parse.urlsplit(raw if '://' in raw else 'http://'+raw)
            result[name]={'set':True,'scheme':p.scheme,'host':p.hostname or '','port':p.port}
        except Exception:
            result[name]={'set':True,'scheme':'','host':'','port':None}
    return result

def certificate_summary():
    result={}
    for name in ENV_CERT:
        value=os.environ.get(name)
        result[name]={'set':bool(value),'path':value if value and not re.search(r'(?i)(key|token|pass|secret)',value) else ('[REDACTED]' if value else ''),'exists':bool(value and Path(value).exists())}
    return result

def runtime_report():
    return {**COMMON,'python_executable':sys.executable,'python_version':sys.version,'os':platform.platform(),'openssl_version':ssl.OPENSSL_VERSION,'default_verify_paths':{k:str(v) if v is not None else None for k,v in ssl.get_default_verify_paths()._asdict().items()},'proxy_environment':proxy_summary(),'certificate_environment':certificate_summary(),'default_socket_timeout':socket.getdefaulttimeout(),'ipv4_available':hasattr(socket,'AF_INET'),'ipv6_available':hasattr(socket,'AF_INET6')}

def probe_dns():
    started=time.perf_counter()
    try:
        infos=socket.getaddrinfo(HOST,PORT,type=socket.SOCK_STREAM); families=sorted(set('IPv6' if x[0]==socket.AF_INET6 else 'IPv4' if x[0]==socket.AF_INET else str(x[0]) for x in infos)); return {'name':'dns','attempts':1,'success':True,'elapsed_s':time.perf_counter()-started,'address_count':len(infos),'address_families':families,'error':None}
    except Exception as exc: return {'name':'dns','attempts':1,'success':False,'elapsed_s':time.perf_counter()-started,'address_count':0,'address_families':[],'error':exception_details(exc,time.perf_counter()-started)}

def probe_tcp():
    started=time.perf_counter(); sock=None
    try:
        sock=socket.create_connection((HOST,PORT),timeout=10); family='IPv6' if sock.family==socket.AF_INET6 else 'IPv4'; return {'name':'tcp','attempts':1,'success':True,'elapsed_s':time.perf_counter()-started,'address_family':family,'error':None}
    except Exception as exc: return {'name':'tcp','attempts':1,'success':False,'elapsed_s':time.perf_counter()-started,'address_family':'','error':exception_details(exc,time.perf_counter()-started)}
    finally:
        if sock is not None: sock.close()

def probe_tls():
    started=time.perf_counter(); raw=None; tls=None; context=ssl.create_default_context()
    try:
        raw=socket.create_connection((HOST,PORT),timeout=10); tls=context.wrap_socket(raw,server_hostname=HOST); cert=tls.getpeercert(); return {'name':'tls','attempts':1,'success':True,'elapsed_s':time.perf_counter()-started,'protocol':tls.version(),'cipher':tls.cipher()[0] if tls.cipher() else '','peer_subject':str(cert.get('subject',''))[:500],'peer_issuer':str(cert.get('issuer',''))[:500],'not_before':cert.get('notBefore',''),'not_after':cert.get('notAfter',''),'hostname_verification':context.check_hostname,'certificate_verification':context.verify_mode==ssl.CERT_REQUIRED,'error':None}
    except Exception as exc: return {'name':'tls','attempts':1,'success':False,'elapsed_s':time.perf_counter()-started,'protocol':'','cipher':'','peer_subject':'','peer_issuer':'','not_before':'','not_after':'','hostname_verification':context.check_hostname,'certificate_verification':context.verify_mode==ssl.CERT_REQUIRED,'error':exception_details(exc,time.perf_counter()-started)}
    finally:
        if tls is not None: tls.close()
        elif raw is not None: raw.close()

def probe_head():
    started=time.perf_counter(); req=urllib.request.Request(HEAD_URL,method='HEAD')
    try:
        with urllib.request.urlopen(req,timeout=15,context=ssl.create_default_context()) as resp: return {'name':'https_head','attempts':1,'success':True,'http_response_received':True,'status':getattr(resp,'status',None),'reason':redact(getattr(resp,'reason','')),'final_host':urllib.parse.urlsplit(resp.geturl()).hostname or '','final_path':urllib.parse.urlsplit(resp.geturl()).path,'content_type':redact(resp.headers.get('Content-Type','')),'redirected':resp.geturl()!=HEAD_URL,'elapsed_s':time.perf_counter()-started,'error':None}
    except urllib.error.HTTPError as exc: return {'name':'https_head','attempts':1,'success':False,'http_response_received':True,'status':exc.code,'reason':redact(exc.reason),'final_host':HOST,'final_path':'/','content_type':redact(exc.headers.get('Content-Type','') if exc.headers else ''),'redirected':False,'elapsed_s':time.perf_counter()-started,'error':exception_details(exc,time.perf_counter()-started)}
    except Exception as exc: return {'name':'https_head','attempts':1,'success':False,'http_response_received':False,'status':None,'reason':'','final_host':HOST,'final_path':'/','content_type':'','redirected':False,'elapsed_s':time.perf_counter()-started,'error':exception_details(exc,time.perf_counter()-started)}

def classify(probes):
    if not probes['dns']['success']: return 'DNS resolution failure'
    if not probes['tcp']['success']: return 'TCP connectivity failure'
    if not probes['tls']['success']: return 'TLS handshake or certificate-validation failure'
    if probes['https_head']['http_response_received']: return 'local HTTPS transport operational'
    return 'HTTP client / urllib / proxy path discrepancy'

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8'); os.chmod(path,0o600)

def run(root=ROOT):
    out=root/'NON_PUBLICATION_SMOKE_TEST'/'score_only_local_transport_diagnosis_20260810';
    if out.exists() and any(out.iterdir()): raise FileExistsError('Refusing to overwrite non-empty output directory')
    out.mkdir(parents=True,exist_ok=True); os.chmod(out,0o700)
    probes={'dns':probe_dns(),'tcp':probe_tcp(),'tls':probe_tls(),'https_head':probe_head()}
    common={**COMMON,'probe_counts':{k:v['attempts'] for k,v in probes.items()},'authenticated_api_requests':0,'classification':classify(probes),'https_server_reached':bool(probes['https_head']['http_response_received']),'authentication_validity_known':False}
    write_json(out/'local_transport_diagnosis.json',{**common,'earliest_failed_layer':next((k for k in ('dns','tcp','tls','https_head') if not probes[k]['success']),'none')})
    write_json(out/'environment_report.json',runtime_report()); write_json(out/'probe_results.json',probes)
    write_json(out/'validation_report.json',{**common,'all_counts_within_limits':all(v<=1 for v in common['probe_counts'].values()),'matlab_simulation_count':0,'full_feedback_started':False})
    return {'output_dir':str(out),'classification':common['classification'],'probe_counts':common['probe_counts']}

if __name__=='__main__': print(json.dumps(run(),indent=2,ensure_ascii=False))
