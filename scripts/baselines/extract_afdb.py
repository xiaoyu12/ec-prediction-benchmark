"""Extract AF-<id>-F1-model_v6.pdb for the listed accessions from the local AFDB Swiss-Prot tarball."""
import gzip, sys, tarfile
from pathlib import Path
ids = set(Path(sys.argv[2]).read_text().split()); out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
found = 0
with tarfile.open(sys.argv[1], "r|") as tar:
    for m in tar:
        acc = m.name.split("-")[1] if m.name.startswith("AF-") else None
        if acc in ids and m.name.endswith("-F1-model_v6.pdb.gz"):
            (out / f"AF-{acc}-F1-model_v6.pdb").write_bytes(gzip.decompress(tar.extractfile(m).read())); found += 1
print("extracted", found, "of", len(ids))
