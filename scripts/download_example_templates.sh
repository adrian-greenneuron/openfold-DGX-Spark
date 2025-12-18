#!/bin/bash
set -e

TARGET_DIR="/opt/openfold/examples/monomer/template_mmcif"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# List of templates used by 6KWC (Medium) and 2Q2K (Short) examples
TEMPLATES="1axk 1f5j 1h0b 1h1a 1h4g 1h8v 1ks4 1ks5 1oa2 1oa3 1oa4 1olr 1pvx 1qh7 1t6g 1te1 1uu5 1xnk 1xyn 1yna 2bw8 2c1f 2dcj 2dck 2dcz 2dfb 2f6b 2jen 2nlr 2qz2 2vg9 2vgd 2vuj 3akq 3b7m 3lb9 3m4f 3mf6 3mf9 3vlb 3wp3 3wp4 3wp5 3wp6 4g6t 4ixl 5ej3 5gm3 5gm4 5gm5 5gv1 5gyc 5gyf 5hxv 5m2d 5tvy 5vqj 6kka 6qe8 1hf9 1psm 2crb 2q2k 3u8v 3zee 4i6p 4zey 5kc1"

echo "Downloading $(echo $TEMPLATES | wc -w) templates to $TARGET_DIR..."

for pdb in $TEMPLATES; do
    if [ ! -f "${pdb}.cif" ]; then
        # Try downloading (using Uppercase for URL, Lowercase for file)
        wget -q "https://files.rcsb.org/download/${pdb^^}.cif" -O "${pdb}.cif" || echo "Failed to download ${pdb}"
    fi
done

echo "Download complete."
chmod 755 $TARGET_DIR/*.cif
