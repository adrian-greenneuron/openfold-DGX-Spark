"""Capture installed runtime identities and the public README benchmark inputs."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--uid', type=int, default=1000)
    parser.add_argument('--gid', type=int, default=1000)
    args = parser.parse_args()
    source = Path('/opt/openfold')
    expected = json.loads((args.audit / 'upstream_source_hashes_251026.json').read_text())
    actual, different, missing = {}, [], []
    for name, value in expected.items():
        path = source / name
        if not path.exists():
            missing.append(name)
            continue
        actual[name] = sha(path)
        if actual[name] != value:
            different.append(name)
            dst = args.audit / 'original_source_differences' / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
    weights = []
    for folder in [source/'openfold_params', source/'openfold_soloseq_params',
                   Path('/root/.cache/torch/hub/checkpoints')]:
        for p in sorted(folder.glob('*')):
            if p.is_file():
                weights.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    resource = source/'openfold/resources/stereo_chemical_props.txt'
    report = {'source_hashes': actual, 'different_source_files': different,
              'missing_source_files': missing, 'weights': weights,
              'stereo_chemical_props_sha256': sha(resource),
              'versions': {d.metadata['Name']: d.version for d in metadata.distributions()
                           if d.metadata.get('Name')}}
    (args.audit/'original_runtime_manifest_251026.json').write_text(json.dumps(report, indent=2)+'\n')
    args.frozen.mkdir(parents=True, exist_ok=True)
    for label, fasta, ident in [('short', source/'tests/test_data/short.fasta', '2q2k_A'),
                                ('medium', source/'examples/monomer/fasta_dir/6kwc.fasta', '6KWC_1')]:
        dest = args.frozen/label
        (dest/'fasta').mkdir(parents=True, exist_ok=True)
        seq = ''.join(line.strip() for line in fasta.read_text().splitlines() if not line.startswith('>'))
        (dest/'fasta'/f'{ident}.fasta').write_text(f'>{ident}\n{seq}\n')
        align_source = source/('tests/test_data/alignments' if label=='short' else 'examples/monomer/alignments')
        shutil.copytree(align_source, dest/'alignments', dirs_exist_ok=True)
        if label=='short':
            top = dest/'alignments'
            (top/ident).mkdir(exist_ok=True)
            if (top/'pdb70_hits.hhr').exists():
                shutil.copy2(top/'pdb70_hits.hhr',top/ident/'pdb70_hits.hhr')
            (top/ident/'hmm_output.sto').unlink(missing_ok=True)
        (dest/'identity.json').write_text(json.dumps({'identifier':ident, 'residues':len(seq)},indent=2)+'\n')
    shutil.copytree(source/'examples/monomer/template_mmcif',args.frozen/'templates',dirs_exist_ok=True)
    input_hashes = {str(p.relative_to(args.frozen)):sha(p) for p in sorted(args.frozen.rglob('*')) if p.is_file()}
    (args.audit/'frozen_inputs_251026.json').write_text(json.dumps(input_hashes,indent=2)+'\n')
    for root in [args.audit,args.frozen]:
        for p in [root,*root.rglob('*')]:
            os.chown(p,args.uid,args.gid)
    print(json.dumps({'source_files':len(actual),'different':different,'missing':missing,
                      'weights':len(weights),'input_files':len(input_hashes)},indent=2))

if __name__=='__main__':
    main()
