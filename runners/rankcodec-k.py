"""
RankCodec-K falsifier: does training a 2-bit K codebook with an explicit RANKING loss
(ListNet listwise CE on softmax(q.K)) beat the query-weighted logit-MSE objective
(A2ATS/OSCAR state of the art) at preserving attention ranking on held-out queries?

Offline, self-contained: capture post-RoPE Q,K from a real model; per (layer, kv-head)
train two 2-bit per-channel scalar K-quantizers (STE) under each objective on calibration
keys/queries; evaluate ranking preservation on a held-out text. If ranking-trained does
not beat logit-MSE-trained above the head-to-head noise, the sliver is dead.
"""
import os, json, math, importlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "8,16,23").split(",")]
LEVELS = 4                       # 2-bit
T_KEYS = int(os.environ.get("T_KEYS", "1024"))
NQ = int(os.environ.get("NQ", "512"))
STEPS = int(os.environ.get("STEPS", "300"))
torch.manual_seed(0)
dev = "cuda"

CALIB = ("The following is a long technical discussion about distributed systems, consensus "
 "protocols, and the trade-offs between consistency and availability. ") * 60
TEST = ("In a separate domain, consider the history of cartography, the projection problem, "
 "and how mapmakers reconciled a curved earth with flat paper over the centuries. ") * 60

cap = []
def _mk(orig):
    def p(q, k, cos, sin, *a, **kw):
        qe, ke = orig(q, k, cos, sin, *a, **kw)
        cap.append((qe.detach(), ke.detach()))
        return qe, ke
    return p
_patched_mods = []
for _name in ["qwen3", "qwen2", "llama"]:
    try:
        _m = importlib.import_module(f"transformers.models.{_name}.modeling_{_name}")
        if hasattr(_m, "apply_rotary_pos_emb"):
            _m.apply_rotary_pos_emb = _mk(_m.apply_rotary_pos_emb)
            _patched_mods.append(_name)
    except Exception:
        pass
print(f"# patched apply_rotary_pos_emb in: {_patched_mods}", flush=True)

print(f"# loading {MODEL}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager")
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager")
model.to(dev)
model.eval()
hd = model.config.head_dim if hasattr(model.config, "head_dim") else model.config.hidden_size // model.config.num_attention_heads
nq_heads = model.config.num_attention_heads
nkv = model.config.num_key_value_heads
g = nq_heads // nkv
print(f"# head_dim={hd} q_heads={nq_heads} kv_heads={nkv} gqa={g}", flush=True)

def capture(text):
    cap.clear()
    ids = tok(text, return_tensors="pt", truncation=True, max_length=T_KEYS).to(dev)
    with torch.no_grad():
        model(**ids)
    out = [(q[0].float(), k[0].float()) for q, k in cap]  # per layer: q[nq,T,hd], k[nkv,T,hd]
    return out

calib = capture(CALIB)
test = capture(TEST)
print(f"# captured layers={len(calib)} (expect >= {max(LAYERS)+1}); calib T={calib[0][1].shape[1]} test T={test[0][1].shape[1]}", flush=True)
assert len(calib) > max(LAYERS), "capture failed (apply_rotary_pos_emb patch did not fire)"

def heads_for(layer, kvh, pack):
    q_all, k_all = pack[layer]              # q[nq,T,hd], k[nkv,T,hd]
    K = k_all[kvh]                          # [T,hd]
    Q = q_all[kvh*g:(kvh+1)*g].reshape(-1, hd)  # [g*T, hd] queries that attend this kv head
    return Q, K

def quantize(K, log_s, c):
    s = log_s.exp()                        # [hd]
    x = K / s                              # [T,hd]
    d = (x.unsqueeze(-1) - c).abs()        # [T,hd,L]
    xq = c[d.argmin(-1)]                   # [T,hd]
    xq = x + (xq - x).detach()             # STE
    return xq * s

def init_params(K):
    log_s = (K.abs().mean(0).clamp_min(1e-4)).log().clone().requires_grad_(True)  # per-channel
    c = torch.linspace(-1.2, 1.2, LEVELS, device=dev).clone().requires_grad_(True) # shared levels
    return log_s, c

def train(Qc, Kc, objective):
    log_s, c = init_params(Kc)
    opt = torch.optim.Adam([log_s, c], lr=0.02)
    scale = 1.0 / math.sqrt(hd)
    Sfp = (Qc @ Kc.t()) * scale            # [Nq,T]
    Pfp = Sfp.softmax(-1)
    for _ in range(STEPS):
        opt.zero_grad()
        Kh = quantize(Kc, log_s, c)
        Sq = (Qc @ Kh.t()) * scale
        if objective == "logit_mse":
            loss = ((Sfp - Sq) ** 2).mean()
        else:  # listnet ranking CE
            loss = -(Pfp * Sq.log_softmax(-1)).sum(-1).mean()
        loss.backward()
        opt.step()
    return log_s.detach(), c.detach()

def naive_mse(Kc):  # per-channel MSE-optimal-ish baseline (no query weighting), fixed init trained on plain MSE
    log_s, c = init_params(Kc)
    opt = torch.optim.Adam([log_s, c], lr=0.02)
    for _ in range(STEPS):
        opt.zero_grad()
        Kh = quantize(Kc, log_s, c)
        loss = ((Kc - Kh) ** 2).mean()
        loss.backward(); opt.step()
    return log_s.detach(), c.detach()

def ndcg_at_k(Sfp, Sq, k=8):
    # gains = softmax(fp) ; rank by quantized scores ; DCG/IDCG
    gains = Sfp.softmax(-1)
    order_q = Sq.argsort(-1, descending=True)[:, :k]
    g_at = torch.gather(gains, 1, order_q)
    disc = 1.0 / torch.log2(torch.arange(2, k+2, device=dev).float())
    dcg = (g_at * disc).sum(-1)
    ideal = gains.sort(-1, descending=True).values[:, :k]
    idcg = (ideal * disc).sum(-1).clamp_min(1e-9)
    return (dcg / idcg).mean().item()

def evaluate(Qt, Kt, log_s, c):
    scale = 1.0 / math.sqrt(hd)
    Sfp = (Qt @ Kt.t()) * scale
    Kh = quantize(Kt, log_s, c)
    Sq = (Qt @ Kh.t()) * scale
    ndcg = ndcg_at_k(Sfp, Sq, 8)
    top1 = (Sfp.argmax(-1) == Sq.argmax(-1)).float().mean().item()
    kl = F.kl_div(Sq.log_softmax(-1), Sfp.softmax(-1), reduction="batchmean").item()
    return ndcg, top1, kl

rows = []
for L in LAYERS:
    for h in range(nkv):
        Qc, Kc = heads_for(L, h, calib)
        Qt, Kt = heads_for(L, h, test)
        if Qc.shape[0] > NQ:
            Qc = Qc[torch.randperm(Qc.shape[0], device=dev)[:NQ]]
        if Qt.shape[0] > NQ:
            Qt = Qt[torch.randperm(Qt.shape[0], device=dev)[:NQ]]
        res = {"layer": L, "kvh": h}
        for name, train_fn in [("logit_mse", lambda: train(Qc, Kc, "logit_mse")),
                               ("ranking",   lambda: train(Qc, Kc, "ranking")),
                               ("naive_mse", lambda: naive_mse(Kc))]:
            ls, c = train_fn()
            ndcg, top1, kl = evaluate(Qt, Kt, ls, c)
            res[name] = {"ndcg8": round(ndcg, 4), "top1": round(top1, 4), "kl": round(kl, 5)}
        rows.append(res)
        print(f"L{L} h{h}: logitMSE ndcg={res['logit_mse']['ndcg8']} top1={res['logit_mse']['top1']} | "
              f"ranking ndcg={res['ranking']['ndcg8']} top1={res['ranking']['top1']} | "
              f"naiveMSE ndcg={res['naive_mse']['ndcg8']} top1={res['naive_mse']['top1']}", flush=True)

# aggregate + verdict
def agg(key, metric):
    return sum(r[key][metric] for r in rows) / len(rows)
print("\n===== AGGREGATE (mean over %d cells) =====" % len(rows), flush=True)
for key in ["naive_mse", "logit_mse", "ranking"]:
    print(f"{key:10s} ndcg8={agg(key,'ndcg8'):.4f} top1={agg(key,'top1'):.4f} kl={agg(key,'kl'):.5f}", flush=True)
d_ndcg = agg("ranking", "ndcg8") - agg("logit_mse", "ndcg8")
d_top1 = agg("ranking", "top1") - agg("logit_mse", "top1")
# per-cell paired deltas (noise band)
dn = [r["ranking"]["ndcg8"] - r["logit_mse"]["ndcg8"] for r in rows]
import statistics as st
sd = st.pstdev(dn) if len(dn) > 1 else 0.0
print(f"\nranking - logit_mse: dNDCG8={d_ndcg:+.4f} (per-cell sd={sd:.4f})  dTop1={d_top1:+.4f}", flush=True)
verdict = "RANKING WINS (beyond noise)" if d_ndcg > 2*sd and d_ndcg > 0.002 else ("TIE / DEAD" if abs(d_ndcg) <= max(2*sd,0.002) else "LOGIT-MSE WINS")
print(f"VERDICT: {verdict}", flush=True)
print("\n===JSON===")
print(json.dumps({"model": MODEL, "layers": LAYERS, "rows": rows,
                  "agg": {k: {m: agg(k,m) for m in ['ndcg8','top1','kl']} for k in ['naive_mse','logit_mse','ranking']},
                  "delta_ndcg": d_ndcg, "delta_top1": d_top1, "percell_sd": sd, "verdict": verdict}))
print("ALLDONE")
