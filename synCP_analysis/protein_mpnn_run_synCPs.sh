#!/bin/bash

# Directory containing the PDB files
PDB_DIR="/home/ubuntu/ProteinMPNN/1wcwA_pdbs/1wcw_synCPs"
# Base output folder
BASE_OUT_DIR="/home/ubuntu/ProteinMPNN/outputs/1wcwA_outputs"

# Make sure base output directory exists
mkdir -p "$BASE_OUT_DIR"

# Loop over all .pdb files in the directory
for pdb_file in "$PDB_DIR"/*.pdb; do
    # Extract just the filename without path
    pdb_name=$(basename "$pdb_file")
    
    # Extract permutation number from filename, e.g., 1bq7_permutation_128.pdb → 128
    perm_num=$(echo "$pdb_name" | grep -oP '(?<=_permutation_)\d+')
    
    # Create output directory for this PDB
    OUT_DIR="$BASE_OUT_DIR/1wcw_permutation_$perm_num"
    mkdir -p "$OUT_DIR"
    
    echo "Running ProteinMPNN on $pdb_name → $OUT_DIR ..."
    
    python /home/ubuntu/ProteinMPNN/protein_mpnn_run.py \
        --pdb_path "$pdb_file" \
        --out_folder "$OUT_DIR" \
        --unconditional_probs_only 1 \
        --num_seq_per_target 1 \
        --batch_size 1
done

echo "All PDBs processed."