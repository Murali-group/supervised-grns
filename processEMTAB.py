import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path


# Heavily inspired by steps detailed in: 
# https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html#Preprocessing

def get_parser() -> argparse.ArgumentParser:
    '''
    :return: an argparse ArgumentParser object for parsing command
        line parameters
    '''
    parser = argparse.ArgumentParser(
        description='Run pathway reconstruction pipeline.')

    parser.add_argument('--emtab', default='E-MTAB-6819', help='E-MTAB id number. Has to be one from: http://www.nxn.se/single-cell-studies/gui')

    parser.add_argument('--org', default='hsapiens', help='Organism name. Has to be either hsapiens or mmusculus.')
    
    
    parser.add_argument('--outPath', default='inputs/datasets/', help='Expression file name for writing output.')
    
    
    parser.add_argument('--hvg', type=int, default=500, help='Number of highly varying genes.')
    
    
    parser.add_argument('--minGenes', type=int, default=200, help='Cutoff for cells. Keeps only cells with at least minGenes expressed.')
    
    parser.add_argument('--minCells', type=int, default=3, help='Cutoff for genes. Keeps only genes which are expressed in at least minCells.')
    
    return parser

def parse_arguments():
    '''
    Initialize a parser and use it to parse the command line arguments
    :return: parsed dictionary of command line arguments
    '''
    parser = get_parser()
    opts = parser.parse_args()

    return opts

def main():
    
    
    opts = parse_arguments()
    
    # Make a temporary data/ directory where scanpy 
    # stores the downloaded data
    # TODO: see if this can be passed as as argument
    
    Path.mkdir(Path('data/'), exist_ok=True)
    eData = sc.datasets.ebi_expression_atlas(opts.emtab)
    
    # once this step is complete, scanpy will read in the data from
    fName = f'data/{opts.emtab}/{opts.emtab}.h5ad'
    
    try:
        adata = sc.read_h5ad(fName)
    except:
        print(f'Unknown error. Check if {fName} exists.\n')
        
    
    # Main preprocessing check//
    
    # check if gene names used are ensembl id: i.e., ENSG or ENSMUSG
    # else, I'm not sure if it is safe to assume anything about it
    # quit, if not ensembl id.
    
    if opts.org == 'hsapiens':
        if not 'ENSG' in adata.var.index[0]:
            print("Gene names in annData are not ensembl ids. Quitting.\n")
            sys.exit()
    elif opts.org == 'mmusculus':
        if not 'ENSMUSG' in adata.var.index[0]:
            print("Gene names in annData are not ensembl ids. Quitting.\n")
            sys.exit()
    else:
        print("Organism name should either be hsapiens or mmusculus. Quitting.\n")
        sys.exit()

    # Keep only unique gene names
    adata.var_names_make_unique()
    

    # Get annotations from biomart
    annot = sc.queries.biomart_annotations(
        opts.org,
        ["ensembl_gene_id", "external_gene_name","start_position", "end_position", "chromosome_name"],
        use_cache=True).set_index("ensembl_gene_id")

    print(f"\nInput raw data has: {str(adata.shape[0])} cells and {str(adata.shape[1])} genes\n")
    
    # Copy annotations
    # Necessary for writing gene names at the end
    adata.var[annot.columns] = annot
    
    # Some minimal filtering.
    sc.pp.filter_cells(adata, min_genes=opts.minGenes)
    sc.pp.filter_genes(adata, min_cells=opts.minCells)

    print(f"\nNumber of cells and genes after initial filtering: {str(adata.shape[0])} cells and {str(adata.shape[1])} genes\n")

    # Step needed for filtering out MT genes
    adata.var['mt'] = adata.var.chromosome_name == 'MT'  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Convert to log counts
    sc.pp.log1p(adata)

    # Compute highly varying genes
    sc.pp.highly_variable_genes(adata, n_top_genes = opts.hvg)
    
    
        
    # Keep only highly varying genes
    adata = adata[:, adata.var.highly_variable]
    print(f"\nAfter applying highly varying genes filter: {str(adata.shape[0])} cells and {str(adata.shape[1])} genes\n")


    # Regress out effects of total counts per cell and the percentage of
    # mitochondrial genes expressed. Scale the data to unit variance.

    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.scale(adata, max_value=10)
    
    
    aDF = adata.to_df()

    # Convert gene names to ensembl ids
    presentGenes = annot.loc[annot.index.intersection(aDF.columns)].external_gene_name.dropna().to_dict()    
    aDFNew = aDF.rename(columns=presentGenes, inplace=False)
    
    aDFNew = aDFNew.T.drop_duplicates()

    aDFNew = aDFNew[~aDFNew.index.duplicated(keep='first')]

    print(f"\nNumber of cells and genes after converting from Ensembl to gene IDs: {str(aDFNew.shape[1])} cells and {str(aDFNew.shape[0])} genes\n")
    
    outPath = os.path.join(opts.outPath, opts.emtab+'-'+str(opts.hvg), opts.emtab+'-'+str(opts.hvg)+'.csv')
    print(f"Writing output to: {outPath}")
    
    os.makedirs(Path('/'.join(outPath.split('/')[:-1])), exist_ok=True)
    aDFNew.to_csv(outPath)

    print("\nDone\n")


if __name__ == '__main__':
    sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    main()
