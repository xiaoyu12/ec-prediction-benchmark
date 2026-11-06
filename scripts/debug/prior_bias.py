"""train.py with the final classifier bias initialised to log(1/num_classes) (prior), to skip the plateau."""
import math, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # scripts/debug/ -> repo root
sys.path.insert(0, str(REPO / "code"))
import torch, train
original = train.build_model
def build_model(name):
    model = original(name)
    last = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.bias.fill_(-math.log(last.out_features))
    print(f"[prior-bias] {name}: bias of {last} set to {-math.log(last.out_features):.3f}", flush=True)
    return model
train.build_model = build_model
sys.argv = ["train.py"] + sys.argv[1:]
raise SystemExit(train.main())
