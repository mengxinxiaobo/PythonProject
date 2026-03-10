import json, re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
prep = root/"Src/runs/prep/swat_T100_S10_H1/features.json"
with open(prep,"r",encoding="utf-8") as f:
    feats = json.load(f)["features"]

def stage_of(name: str):
    m = re.search(r"(\d{3})", name)
    if not m:
        return None
    code = int(m.group(1))
    return code // 100  # 101->1, 202->2, 503->5

groups = {}
for idx, name in enumerate(feats):
    s = stage_of(name)
    groups.setdefault(s, []).append(idx)

subgraphs = {}
for s, nodes in sorted(groups.items()):
    if s is None:
        continue
    subgraphs[f"S{s}"] = {"nodes": nodes, "node_names":[feats[i] for i in nodes], "size": len(nodes)}

out = root/"Src/runs/subgraphs/subgraphs_stage.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(subgraphs, f, ensure_ascii=False, indent=2)

print("saved:", out)
print({k:v["size"] for k,v in subgraphs.items()})