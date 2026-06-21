import subprocess, re, json, sys, time, random, os, threading
# decoy-bench-f0: controles (Boundary V opt-out, per-case dump, prompt-seed jitter)
# + metricas continuas (erro por-digito, edit-distance) para F0 (forma vs regua).
# Compativel com decoy-bench.py (mesmas env DEPTHS/CONFIGS/NPOS/RATIO/CTX/MODEL/TAG).
# Novas env: PROMPT_SEED (default 42, re-seed da geracao do prompt p/ jitter de instancia),
#            DUMP (path JSONL p/ dump per-case). TURBO_LAYER_ADAPTIVE propaga ao build.
D=os.environ.get('LCT','/home/felipe/llama-cpp-turboquant/build')
BIN=f"{D}/bin/llama-completion"
MODEL=os.environ.get('MODEL',"/mnt/c/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
TAG=os.environ.get('TAG','llama31-8b-f0')
ENV=dict(os.environ, LD_LIBRARY_PATH=f"{D}/bin:{D}/src:/usr/local/cuda-13.0/lib64", PATH="/usr/local/cuda-13.0/bin:"+os.environ.get('PATH',''))
DEPTHS=[16384,20480,24576,28672,32768]
CONFIGS=['f16','turbo3','turbo2']
POSITIONS=[round(0.03+i*(0.94-0.03)/(8-1),3) for i in range(8)]
if os.environ.get('DEPTHS'): DEPTHS=[int(x) for x in os.environ['DEPTHS'].split(',')]
if os.environ.get('CONFIGS'): CONFIGS=os.environ['CONFIGS'].split(',')
if os.environ.get('NPOS'):
    k=int(os.environ['NPOS']); POSITIONS=[round(0.03+i*(0.94-0.03)/(k-1),3) for i in range(k)]
RATIO=float(os.environ.get('RATIO','2.4'))
CTXABS=os.environ.get('CTX')
PROMPT_SEED=int(os.environ.get('PROMPT_SEED','42'))
DUMP=os.environ.get('DUMP')
WORDS=['ALPHA','BRAVO','DELTA','ECHO','FOXTROT','GOLF','HOTEL','INDIA','JULIET','KILO','LIMA','MIKE']
random.seed(PROMPT_SEED)
def mkcode(w): return f"{random.randint(1000,9999)}-{w}"
def filler(nchars):
    out=[]; n=0; i=0
    while n<nchars:
        i+=1; s=f"Log entry {i}: sensor {random.randint(100,999)} read {random.randint(1000,9999)} at node {random.randint(10,99)} cycle {random.randint(1,500)}. "; out.append(s); n+=len(s)
    return ''.join(out)
UNITS=['Vega','Lyra','Cygnus','Draco','Hydra','Perseus']
def build(depth,pos):
    ws=random.sample(WORDS,5); us=random.sample(UNITS,4)
    C=mkcode(ws[0]); dcodes=[mkcode(w) for w in ws[1:]]
    canon=f"The secret access code for unit Orion is {C}. "
    dfracs=[0.12,0.40,0.62,0.85]
    decoys=[(dfracs[i], f"The secret access code for unit {us[i]} is {dcodes[i]}. ") for i in range(4)]
    items=sorted(decoys+[(pos,canon)], key=lambda x:x[0])
    hay=filler(int(depth*RATIO)); res=[]; last=0
    for frac,txt in items:
        cut=max(int(len(hay)*frac), last); res.append(hay[last:cut]); res.append(txt); last=cut
    res.append(hay[last:]); full=''.join(res)
    q="QUESTION: What is the secret access code for unit Orion? Reply with ONLY the code, nothing else.\nANSWER:"
    prompt=f"You are given a long log listing secret access codes for several units. Read it and answer the question for the unit asked.\n\n{full}\n\n{q}"
    return prompt, C, dcodes
def answer_part(out):
    i=out.rfind("ANSWER:"); return out[i+7:] if i>=0 else out
def extract(ap):
    m=re.search(r'\d{4}-[A-Z]+', ap); return m.group(0) if m else None
def lev(a,b):
    a=a or ''; b=b or ''
    if a==b: return 0
    m,n=len(a),len(b)
    if m==0: return n
    if n==0: return m
    prev=list(range(n+1))
    for i in range(1,m+1):
        cur=[i]+[0]*n
        for j in range(1,n+1):
            cur[j]=min(prev[j]+1, cur[j-1]+1, prev[j-1]+(a[i-1]!=b[j-1]))
        prev=cur
    return prev[n]
def digit_err(ans,C):
    # erro por-digito no bloco DDDD; max 4 se ausente/malformado
    cd=re.match(r'(\d{4})-',C);
    if not cd: return None,False
    ctgt=cd.group(1)
    if not ans: return 4,False
    ad=re.match(r'(\d{4})-',ans)
    if not ad: return 4,False
    asrc=ad.group(1)
    de=sum(1 for x,y in zip(asrc,ctgt) if x!=y)
    word_ok=(ans.split('-',1)[-1]==C.split('-',1)[-1]) if ('-' in ans and '-' in C) else False
    return de, word_ok
def adaptive_from_err(err):
    # captura a linha de log do modo layer-adaptive / boundary V
    for pat in [r'InnerQ[^\n]*', r'Boundary V[^\n]*', r'layer-adaptive mode[^\n]*', r'auto-enabled[^\n]*', r'asymmetric[^\n]*']:
        m=re.search(pat,err)
        if m: return m.group(0).strip()
    return ''
def gpu_peak(stop,box):
    mx=0
    while not stop.is_set():
        try:
            u=int(subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=4).stdout.split("\n")[0]); mx=max(mx,u)
        except: pass
        time.sleep(0.5)
    box['peak']=mx
def run(prompt,kv,ctx):
    # prompt via arquivo (-f): prompts longos (>~50k) estouram ARG_MAX no -p.
    # try/finally garante que a thread de GPU sempre para (senao OSError pendura o processo).
    import tempfile
    stop=threading.Event(); box={'peak':0}; th=threading.Thread(target=gpu_peak,args=(stop,box)); th.start()
    pf=None
    try:
        fd,pf=tempfile.mkstemp(suffix='.txt',dir='/mnt/c/ops')
        with os.fdopen(fd,'w') as f: f.write(prompt)
        ctk=os.environ.get('CTK') or kv; ctv=os.environ.get('CTV') or kv
        try:
            r=subprocess.run([BIN,"-m",MODEL,"-ngl","99","-c",str(ctx),"-n","24","--temp","0","--seed","1","-st","--simple-io","-fa","on","-ctk",ctk,"-ctv",ctv,"-f",pf],capture_output=True,text=True,timeout=900,env=ENV)
            out,err,rc=r.stdout,r.stderr,r.returncode
        except subprocess.TimeoutExpired: out,err,rc='','TIMEOUT',124
        except OSError as e: out,err,rc='',f'OSERR:{e}',125
    finally:
        stop.set(); th.join()
        if pf and os.path.exists(pf):
            try: os.remove(pf)
            except OSError: pass
    return out,err,rc,box['peak']
def dec_tps(err):
    for m in re.finditer(r'(prompt )?eval time =.*?([\d.]+) tokens per second',err):
        if not m.group(1): return float(m.group(2))
    return None
dumpf=open(DUMP,'a') if DUMP else None
rows=[]
print(f"# PROMPT_SEED={PROMPT_SEED} TURBO_LAYER_ADAPTIVE={os.environ.get('TURBO_LAYER_ADAPTIVE','(unset->auto)')} RATIO={RATIO} DUMP={DUMP}",flush=True)
for depth in DEPTHS:
    ctx=int(CTXABS) if CTXABS else depth+2048; cases=[build(depth,p) for p in POSITIONS]
    for kv in CONFIGS:
        ret=0; dec=0; decoy=0; tot=0; tps=[]; peaks=[]; derrs=[]; levs=[]; malformed=0; adapt=''
        for (pr,C,dlist),pos in zip(cases,POSITIONS):
            out,err,rc,peak=run(pr,kv,ctx)
            ap=answer_part(out); ans=extract(ap); retrieved=(C in ap); correct=(ans==C); decoyhit=(ans in dlist)
            de,word_ok=digit_err(ans,C); ld=lev(ans,C); mal=(ans is None) or (de==4 and not retrieved)
            ret+=retrieved; dec+=correct; decoy+=decoyhit; tot+=1
            if de is not None: derrs.append(de)
            levs.append(ld);
            if mal: malformed+=1
            if not adapt: adapt=adaptive_from_err(err)
            ts=dec_tps(err)
            if ts:tps.append(ts)
            if peak:peaks.append(peak)
            rec={'tag':TAG,'pseed':PROMPT_SEED,'adaptive_env':os.environ.get('TURBO_LAYER_ADAPTIVE','auto'),'depth':depth,'kv':kv,'pos':pos,'C':C,'decoys':dlist,'ans':ans,'ap':ap[:140],'retrieved':retrieved,'correct':correct,'decoyhit':decoyhit,'digit_err':de,'word_ok':word_ok,'lev':ld,'rc':rc,'adapt_log':adaptive_from_err(err)}
            if dumpf: dumpf.write(json.dumps(rec)+"\n"); dumpf.flush()
            print(f"depth={depth} kv={kv} pos={pos} correct={correct} retrieved={retrieved} decoyhit={decoyhit} digit_err={de} lev={ld} ans={ans} C={C} rc={rc}",flush=True)
        cell={'tag':TAG,'pseed':PROMPT_SEED,'adaptive_env':os.environ.get('TURBO_LAYER_ADAPTIVE','auto'),'depth':depth,'kv':kv,'n':tot,'retrieval':round(ret/tot,3),'decision_acc':round(dec/tot,3),'decoy_rate':round(decoy/tot,3),'mean_digit_err':round(sum(derrs)/len(derrs),3) if derrs else None,'mean_lev':round(sum(levs)/len(levs),3) if levs else None,'malformed_rate':round(malformed/tot,3),'adapt':adapt,'tok_s':round(sum(tps)/len(tps),1) if tps else None,'peak_vram_mib':max(peaks) if peaks else None}
        rows.append(cell)
        print(f">>> CELL {TAG} pseed={PROMPT_SEED} depth={depth} kv={kv} decision={cell['decision_acc']} decoy={cell['decoy_rate']} mDigitErr={cell['mean_digit_err']} mLev={cell['mean_lev']} malformed={cell['malformed_rate']} adapt='{adapt}' tok/s={cell['tok_s']} vram={cell['peak_vram_mib']}",flush=True)
if dumpf: dumpf.close()
print("\n===== SUMMARY ====="); print(json.dumps(rows,indent=1))
