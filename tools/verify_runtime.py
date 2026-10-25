"""Compare a rebuilt installation with a captured original runtime manifest."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-source-difference',action='append',default=[])
    p.add_argument('--allow-version-difference',action='append',default=[])
    args=p.parse_args();ref=json.loads(args.reference.read_text())
    versions={d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata.get('Name')}
    report={'source_mismatches':[name for name,value in ref['source_hashes'].items() if sha(Path('/opt/openfold')/name)!=value],
            'weight_mismatches':[r['path'] for r in ref['weights'] if sha(Path(r['path']))!=r['sha256']],
            'version_mismatches':{k:{'original':v,'rebuilt':versions.get(k)} for k,v in ref['versions'].items() if versions.get(k)!=v},
            'extra_distributions':{k:v for k,v in versions.items() if k not in ref['versions']},
            'source_files_checked':len(ref['source_hashes']),'weights_checked':len(ref['weights']),
            'distributions_checked':len(ref['versions']),
            'stereo_chemical_props_match':sha(Path('/opt/openfold/openfold/resources/stereo_chemical_props.txt'))==ref['stereo_chemical_props_sha256']}
    report['allowed_source_differences']=args.allow_source_difference
    report['allowed_version_differences']=args.allow_version_difference
    report['matches']=not (set(report['source_mismatches'])-set(args.allow_source_difference) or report['weight_mismatches'] or set(report['version_mismatches'])-set(args.allow_version_difference)) and report['stereo_chemical_props_match']
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['matches']:raise SystemExit(1)

if __name__=='__main__':main()
