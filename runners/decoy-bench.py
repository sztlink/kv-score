import subprocess, re, json, sys, time, random, os, threading
D=os.environ.get('LCT','/home/felipe/llama-cpp-turboquant/build')
BIN=f"{D}/bin/llama-completion"
MODEL=os.environ.get('MODEL',"/mnt/c/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
TAG=os.environ.get('TAG','llama31-8b')
ENV=dict(os.environ, LD_LIBRARY_PATH=f"{D}/bin:{D}/src:/usr/local/cuda-13.0/lib64", PATH="/usr/local/cuda-13.0/bin:"+os.environ.get('PATH',''))
QUICK='--quick' in sys.argv
DEPTHS=[8192] if QUICK else [4096,8192,16384,32768]
CONFIGS=['f16','turbo2'] if QUICK else ['f16','turbo4','turbo3','turbo2']
POSITIONS=[0.3,0.8] if QUICK else [0.05,0.3,0.55,0.8,0.95]
if os.environ.get('DEPTHS'): DEPTHS=[int(x) for x in os.environ['DEPTHS'].split(',')]
if os.environ.get('CONFIGS'): CONFIGS=os.environ['CONFIGS'].split(',')
if os.environ.get('NPOS'):
    k=int(os.environ['NPOS']); POSITIONS=[round(0.03+i*(0.94-0.03)/(k-1),3) for i in range(k)]
RATIO=float(os.environ.get('RATIO','2.4'))
CTXABS=os.environ.get('CTX')
WORDS=['ALPHA','BRAVO','DELTA','ECHO','FOXTROT','GOLF','HOTEL','INDIA','JULIET','KILO','LIMA','MIKE']
random.seed(42)
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
def gpu_peak(stop,box):
    mx=0
    while not stop.is_set():
        try:
            u=int(subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=4).stdout.split("\n")[0]); mx=max(mx,u)
        except: pass
        time.sleep(0.5)
    box['peak']=mx
def run(prompt,kv,ctx):
    stop=threading.Event(); box={'peak':0}; th=threading.Thread(target=gpu_peak,args=(stop,box)); th.start()
    try:
        r=subprocess.run([BIN,"-m",MODEL,"-ngl","99","-c",str(ctx),"-n","24","--temp","0","--seed","1","-st","--simple-io","-fa","on","-ctk",kv,"-ctv",kv,"-p",prompt],capture_output=True,text=True,timeout=900,env=ENV)
        out,err,rc=r.stdout,r.stderr,r.returncode
    except subprocess.TimeoutExpired: out,err,rc='','TIMEOUT',124
    stop.set(); th.join()
    return out,err,rc,box['peak']
def dec_tps(err):
    for m in re.finditer(r'(prompt )?eval time =.*?([\d.]+) tokens per second',err):
        if not m.group(1): return float(m.group(2))
    return None
rows=[]
for depth in DEPTHS:
    ctx=int(CTXABS) if CTXABS else depth+2048; cases=[build(depth,p) for p in POSITIONS]
    for kv in CONFIGS:
        ret=0; dec=0; decoy=0; tot=0; tps=[]; peaks=[]
        for (pr,C,dlist),pos in zip(cases,POSITIONS):
            out,err,rc,peak=run(pr,kv,ctx)
            ap=answer_part(out); ans=extract(ap); retrieved=(C in ap); correct=(ans==C); decoyhit=(ans in dlist)
            ret+=retrieved; dec+=correct; decoy+=decoyhit; tot+=1
            ts=dec_tps(err)
            if ts:tps.append(ts)
            if peak:peaks.append(peak)
            dbg=repr(ap[:60]) if QUICK else ''
            print(f"depth={depth} kv={kv} pos={pos} retrieved={retrieved} correct={correct} decoyhit={decoyhit} ans={ans} C={C} rc={rc} ap={dbg}",flush=True)
        rows.append({'tag':TAG,'depth':depth,'kv':kv,'n':tot,'retrieval':round(ret/tot,3),'decision_acc':round(dec/tot,3),'decoy_rate':round(decoy/tot,3),'tok_s':round(sum(tps)/len(tps),1) if tps else None,'peak_vram_mib':max(peaks) if peaks else None})
        r=rows[-1]
        print(f">>> CELL {TAG} depth={depth} kv={kv} retrieval={r['retrieval']} decision={r['decision_acc']} decoy={r['decoy_rate']} tok/s={r['tok_s']} vram={r['peak_vram_mib']}",flush=True)
print("\n===== SUMMARY ====="); print(json.dumps(rows,indent=1))
