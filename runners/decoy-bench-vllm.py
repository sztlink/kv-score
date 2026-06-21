import json, re, os, sys, time, random, urllib.request
# decoy-at-depth para a API OpenAI do vLLM (F1: KVarN vs TurboQuant vs fp16, mesmo modelo).
# Mesma construcao de prompt do decoy-bench-f0.py; run() faz POST /v1/chat/completions.
# Le usage.prompt_tokens (profundidade real). Qwen3 thinking DESLIGADO.
ENDPOINT=os.environ.get('ENDPOINT','http://localhost:8001/v1/chat/completions')
MODEL=os.environ.get('MODEL','spikemodel')
TAG=os.environ.get('TAG','vllm')
DEPTHS=[int(x) for x in os.environ.get('DEPTHS','8192,16384,32768').split(',')]
NPOS=int(os.environ.get('NPOS','8'))
POSITIONS=[round(0.03+i*(0.94-0.03)/(NPOS-1),3) for i in range(NPOS)]
RATIO=float(os.environ.get('RATIO','3.6'))
PROMPT_SEED=int(os.environ.get('PROMPT_SEED','42'))
DUMP=os.environ.get('DUMP')
WORDS=['ALPHA','BRAVO','DELTA','ECHO','FOXTROT','GOLF','HOTEL','INDIA','JULIET','KILO','LIMA','MIKE']
UNITS=['Vega','Lyra','Cygnus','Draco','Hydra','Perseus']
random.seed(PROMPT_SEED)
def mkcode(w): return f"{random.randint(1000,9999)}-{w}"
def filler(nchars):
    out=[]; n=0; i=0
    while n<nchars:
        i+=1; s=f"Log entry {i}: sensor {random.randint(100,999)} read {random.randint(1000,9999)} at node {random.randint(10,99)} cycle {random.randint(1,500)}. "; out.append(s); n+=len(s)
    return ''.join(out)
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
    q="QUESTION: What is the secret access code for unit Orion? Reply with ONLY the code, nothing else."
    prompt=f"You are given a long log listing secret access codes for several units. Read it and answer the question for the unit asked.\n\n{full}\n\n{q} /no_think"
    return prompt, C, dcodes
def extract(txt):
    m=re.search(r'\d{4}-[A-Z]+', txt); return m.group(0) if m else None
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
    cd=re.match(r'(\d{4})-',C)
    if not cd: return None,False
    ctgt=cd.group(1)
    if not ans: return 4,False
    ad=re.match(r'(\d{4})-',ans)
    if not ad: return 4,False
    de=sum(1 for x,y in zip(ad.group(1),ctgt) if x!=y)
    word_ok=(ans.split('-',1)[-1]==C.split('-',1)[-1]) if ('-' in ans and '-' in C) else False
    return de, word_ok
def run(prompt):
    payload=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt}],
        "temperature":0,"max_tokens":40,"seed":1,
        "chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(ENDPOINT,data=payload,headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=600) as r:
            d=json.loads(r.read())
        txt=d['choices'][0]['message']['content']; ptoks=d.get('usage',{}).get('prompt_tokens')
        return txt, ptoks, ''
    except Exception as e:
        body=''
        try: body=e.read().decode()[:200]
        except Exception: pass
        return '', None, f"{type(e).__name__}:{e} {body}"
dumpf=open(DUMP,'a') if DUMP else None
rows=[]
print(f"# ENDPOINT={ENDPOINT} MODEL={MODEL} TAG={TAG} RATIO={RATIO} PROMPT_SEED={PROMPT_SEED} DEPTHS={DEPTHS} NPOS={NPOS}",flush=True)
for depth in DEPTHS:
    cases=[build(depth,p) for p in POSITIONS]
    ret=0; dec=0; decoy=0; tot=0; derrs=[]; levs=[]; malformed=0; ptoks_seen=[]; errs=0
    for (pr,C,dlist),pos in zip(cases,POSITIONS):
        txt,ptoks,err=run(pr)
        if err: errs+=1
        ans=extract(txt); retrieved=(C in txt); correct=(ans==C); decoyhit=(ans in dlist)
        de,word_ok=digit_err(ans,C); ld=lev(ans,C); mal=(ans is None)
        ret+=retrieved; dec+=correct; decoy+=decoyhit; tot+=1
        if de is not None: derrs.append(de)
        levs.append(ld)
        if mal: malformed+=1
        if ptoks: ptoks_seen.append(ptoks)
        rec={'tag':TAG,'pseed':PROMPT_SEED,'depth':depth,'pos':pos,'C':C,'ans':ans,'txt':txt[:140],
             'retrieved':retrieved,'correct':correct,'decoyhit':decoyhit,'digit_err':de,'word_ok':word_ok,'lev':ld,'ptoks':ptoks,'err':err[:120]}
        if dumpf: dumpf.write(json.dumps(rec)+"\n"); dumpf.flush()
        print(f"depth={depth} pos={pos} correct={correct} retrieved={retrieved} decoyhit={decoyhit} digit_err={de} lev={ld} ans={ans} C={C} ptoks={ptoks} err={err[:60]}",flush=True)
    cell={'tag':TAG,'pseed':PROMPT_SEED,'depth':depth,'n':tot,'retrieval':round(ret/tot,3),'decision_acc':round(dec/tot,3),
          'decoy_rate':round(decoy/tot,3),'mean_digit_err':round(sum(derrs)/len(derrs),3) if derrs else None,
          'mean_lev':round(sum(levs)/len(levs),3) if levs else None,'malformed_rate':round(malformed/tot,3),
          'median_ptoks':sorted(ptoks_seen)[len(ptoks_seen)//2] if ptoks_seen else None,'errors':errs}
    rows.append(cell)
    print(f">>> CELL {TAG} depth={depth} decision={cell['decision_acc']} retrieval={cell['retrieval']} decoy={cell['decoy_rate']} mLev={cell['mean_lev']} malformed={cell['malformed_rate']} ptoks={cell['median_ptoks']} errs={errs}",flush=True)
if dumpf: dumpf.close()
print("\n===== SUMMARY ====="); print(json.dumps(rows,indent=1))
