"""Create compact measurements, Markdown tables and a PNG from this comparison."""
import argparse
import json
from pathlib import Path
import statistics
import shutil

def median(rows,key):return statistics.median(r[key] for r in rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--docs',type=Path,required=True)
    args=p.parse_args();args.docs.mkdir(parents=True,exist_ok=True)
    data=args.docs/'data';data.mkdir(exist_ok=True)
    conditions=['original','rebuilt','modern'];labels=['Original','Rebuilt original','Modernised']
    measurements={c:json.loads((args.results/'benchmarks'/c/'measurements.json').read_text()) for c in conditions}
    for rows in measurements.values():
        assert len(rows)==30 and all(r['completed'] for r in rows)
    (data/'benchmark_measurements_251026.json').write_text(json.dumps(measurements,indent=2)+'\n')
    for name in ['protocol_251026.json','rebuilt_runtime_comparison_251026.json','modern_runtime_comparison_251026.json',
                 'modern_runtime_qualification_251026.json','original_rebuilt_structures_251026.json',
                 'original_modern_structures_251026.json']:
        shutil.copy2(args.results/'audit'/name,data/name)
    rows=[]
    for protein,residues in [('short',73),('medium',191)]:
        for task in ['standard','soloseq']:
            for relax in [False,True]:
                row={'protein':protein,'residues':residues,'mode':task,'relaxation':relax}
                for condition in conditions:
                    selected=[r for r in measurements[condition] if not r['warmup'] and r['protein']==protein and r['task']==task and r['relaxation']==relax]
                    embedding=[r for r in measurements[condition] if not r['warmup'] and r['protein']==protein and r['task']=='embedding']
                    wall=[r['wall_seconds']+(embedding[i]['wall_seconds'] if task=='soloseq' else 0) for i,r in enumerate(selected)]
                    phase=selected+(embedding if task=='soloseq' else [])
                    row[condition]={'workflow_wall_seconds':statistics.median(wall),'inference_process_wall_seconds':median(selected,'wall_seconds'),
                                    'peak_cuda_allocated_bytes':max(r['peak_cuda_allocated_bytes'] for r in phase),
                                    'peak_cuda_reserved_bytes':max(r['peak_cuda_reserved_bytes'] for r in phase),
                                    'peak_main_process_rss_bytes':max(r['peak_main_process_rss_bytes'] for r in phase),
                                    'sampled_system_nonavailable_peak_bytes':max(r['sampled_system_nonavailable_peak_bytes'] for r in phase)}
                rows.append(row)
    (data/'benchmark_summary_251026.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# OpenFold rebuild and dependency modernisation — 25 October 2026','',
           'The original deployment was rebuilt from public downloads and compared with a separate dependency-modernised image on DGX Spark. All three conditions completed the eight published benchmark configurations, with one warm-up and two measured runs per configuration. Embedding generation was measured separately for each protein and condition.','',
           'The input FASTA files contain **73 and 191 residues**. The previous README labels of 35 and 152 were incorrect. Historical timings are retained as historical observations; this is a newly controlled comparison.','',
           '## Runtime','',
           'Seconds, medians of two runs. SoloSeq workflow time includes a separately timed embedding-generation process plus inference/relaxation. It does not include initial software/model downloads or image building. All conditions used identical saved embeddings for the controlled inference comparison; separately generated embeddings were checked for agreement.','',
           '| Residues | Mode | Relaxation | Original | Rebuilt original | Modernised |',
           '|---:|---|:---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['residues']} | {'Standard' if row['mode']=='standard' else 'SoloSeq'} | {'Yes' if row['relaxation'] else 'No'} | "+' | '.join(f"{row[c]['workflow_wall_seconds']:.2f}" for c in conditions)+' |')
    lines+=['','## Peak memory','',
            'GB decimal. Each cell is **CUDA allocated / main-process RSS**. Values are the maximum observed across the two measured runs and, for SoloSeq, both sequential workflow stages. These measures overlap on unified-memory hardware and must not be added. RSS is process memory, not total machine RAM; the raw records also contain CUDA reserved memory and sampled whole-system memory.','',
            '| Residues | Mode | Relaxation | Original | Rebuilt original | Modernised |',
            '|---:|---|:---:|---:|---:|---:|']
    for row in rows:
        cells=[f"{row[c]['peak_cuda_allocated_bytes']/1e9:.2f} / {row[c]['peak_main_process_rss_bytes']/1e9:.2f}" for c in conditions]
        lines.append(f"| {row['residues']} | {'Standard' if row['mode']=='standard' else 'SoloSeq'} | {'Yes' if row['relaxation'] else 'No'} | "+' | '.join(cells)+' |')
    lines+=['','## Prediction agreement and limitations','',
            '- All 16 measured unrelaxed PDB outputs matched the original byte-for-byte in the rebuilt image, and all 16 matched in the modernised image. This is output agreement on these fixtures, not a new biological-accuracy benchmark.',
            '- GPU relaxation was not byte-repeatable within the original image. The low-confidence short Standard prediction showed particularly large relaxation variability: the original’s two measured relaxed structures differed by about 5.18 Å Cα RMSD and had Cα lDDT 0.436. The same unrelaxed structure entered minimisation each time.',
            '- Fixing preparation seeds and enabling deterministic CUDA forces did not make GPU relaxation byte-repeatable in the focused follow-up. The documented deterministic verification therefore explicitly skips relaxation. Normal inference defaults are unchanged.',
            '- Relaxation time is variable; the short Standard relaxation result must not be interpreted as a general runtime regression or improvement from only two repetitions. Paired and within-image structural measurements are included in the JSON records.',
            '- The modernised build uses the same current upstream OpenFold source revision and the same model weights as the original. It updates OpenMM to published 8.6.1 CUDA 13 ARM64 wheels and DeepSpeed to 0.19.8; it is not a newer trained model. DeepSpeed attention remains disabled in these preset/default comparisons.',
            '- Inputs, alignments, templates, seeds and precision were fixed. The expandable CUDA allocator was set before CUDA initialisation. Peak CUDA memory and process RSS are recorded independently; the historical README whole-machine memory numbers use a different measurement convention.',
            '- The tests cover these two monomers only. Maximum supported size, multimer performance, alternative attention backends and accuracy against experimental structures were not established.','',
            '![Runtime and CUDA allocation](assets/openfold_benchmarks_251026.png)','',
            'Raw measurements and comparison records are in [data/](data/). Build identities, download checksums, deterministic verification instructions and the scope of each check are described in [reproducibility.md](reproducibility.md).','']
    (args.docs/'openfold_benchmarks_251026.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    assets=args.docs/'assets';assets.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
    y=np.arange(len(rows));colors=['#555555','#2878b5','#009e73'];height=.23
    names=[f"{r['residues']} aa · {'Standard' if r['mode']=='standard' else 'SoloSeq'} · {'relax' if r['relaxation'] else 'no relax'}" for r in rows]
    for k,c in enumerate(conditions):
        for ax,metric,scale in [(axes[0],'workflow_wall_seconds',1),(axes[1],'peak_cuda_allocated_bytes',1e9)]:
            ax.barh(y+(k-1)*height,[r[c][metric]/scale for r in rows],height=height,color=colors[k],label=labels[k])
    axes[0].set_yticks(y,names);axes[0].invert_yaxis()
    axes[0].set_xlabel('Complete workflow (seconds; median of two runs)')
    axes[1].set_xlabel('Peak CUDA allocation (GB; includes embedding stage)')
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(loc='lower right',fontsize=9)
    fig.suptitle('OpenFold on DGX Spark: matched models and inputs',fontweight='bold')
    fig.tight_layout();fig.savefig(assets/'openfold_benchmarks_251026.png',dpi=160);plt.close(fig)
    print(json.dumps({'cases':len(rows),'conditions':len(conditions),'runs':sum(map(len,measurements.values())),'docs':str(args.docs)},indent=2))

if __name__=='__main__':main()
