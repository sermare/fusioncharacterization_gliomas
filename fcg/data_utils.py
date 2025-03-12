# data_utils.py

from fcg.common_imports import *

########################################################################
# Data Loading & Basic Cleaning
########################################################################

def load_data_from_directory(directory, pattern="*fusions.tsv", column=-1):
    """
    Recursively load and concatenate files matching a glob pattern
    from a given directory, adding a new column with the file path.

    Parameters:
        directory (str): The root directory where the files are located.
        pattern (str): The glob pattern to match file names (default: "*_fusions.tsv").
        column (int): The part of the file path to extract (default: last directory name).

    Returns:
        pandas.DataFrame: A concatenated DataFrame from all matched files.
    """
    file_paths = glob.glob(f"{directory}/**/{pattern}", recursive=True)
    dfs = []
    
    for fp in file_paths:
        try:
            df = pd.read_csv(fp, sep='\t')
            df['file_path'] = fp.split('/')[column].replace(f'{pattern}', '')
            dfs.append(df)
        except Exception as e:
            print(f"Error processing {fp}: {e}")
    
    return pd.concat(dfs, ignore_index=True)

def preprocess_dataframe(df):
    """
    Perform common preprocessing tasks on a DataFrame.
    Currently, this function drops duplicate rows.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame to preprocess.
    
    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    df = df.drop_duplicates()
    # Additional cleaning steps (e.g., renaming columns, handling missing data)
    # can be added here.

    return df


def process_tables(tables, table = 'arriba'):
    tables['file_path'] = tables['file_path'].str.upper()

    tables['patient'] = [extract_p_number(s) for s in  tables['file_path'] ]

    tables['SF#'] = [extract_sf_number(s) for s in  tables['file_path'] ]

    if table == 'arriba':
        tables['fusion_gene'] = tables['#gene1'] + '/' + tables['gene2']
        # return tables
    else:
        tables['fusion_gene'] = tables['#FusionName'].str.replace('--','/')
    return tables

########################################################################
# Helper functions for Annotations
########################################################################

def extract_p_number(input_string):

    specific_mappings = {
    'SF12069': 'P470',
    'SF12054': 'P469',
    'SF10968': 'P345',
    'SF11056': 'P388',
    'SF11064': 'P236'
    }


    match = re.search(r'P\d{3}', input_string)
    if match:
        return match.group()
    
    match = re.search(r'P\d{2}', input_string)
    if match:
        return match.group()
    
    for key in specific_mappings:
        if key in input_string:
            return specific_mappings[key]

    match = re.search(r'SF\d{5}', input_string)
    if match:
        return match.group()
    
    return None

def extract_sf_number(input_string):

    match = re.search(r'SF\d{5}', input_string)
    if match:
        return match.group()
    
    return None

def process_fusion_purity_data(final_df, purity_estimates):
    """
    Processes fusion data by extracting version numbers, constructing join keys, 
    and merging with purity estimates.

    Parameters:
        final_df (pd.DataFrame): DataFrame containing fusion gene data with 'SF#' and 'file_path_y'.
        purity_estimates (pd.DataFrame): DataFrame containing purity estimates with 'SF#unique'.

    Returns:
        pd.DataFrame: Merged and processed DataFrame.
    """
    
    # Helper function to extract version from SF#unique
    def extract_sf_unique_version(sf_unique_str):
        """
        Extracts version information from SF#unique column.
        Example: 'SF4454v3' -> 'v3'; 'SF12827-v9-2' -> 'v9-2'
        """
        match = re.search(r'^SF\d+(.*)$', sf_unique_str)
        if not match:
            return None
        remainder = match.group(1)
        match2 = re.search(r'v[0-9A-Za-z-]+', remainder)
        return match2.group(0) if match2 else None

    # Extract version from SF#unique in purity_estimates
    purity_estimates['version_extracted'] = purity_estimates['SF#unique'].astype(str).apply(extract_sf_unique_version)

    # Helper function to extract version from file_path_y in final_df
    def extract_file_path_version(fp):
        """
        Extracts version information from the file path.
        Example: 'P533SF12827-V9_S290_' -> 'v9'
        """
        match = re.search(r'V(\d+(-\d+)?)', fp, flags=re.IGNORECASE)
        return f"v{match.group(1)}" if match else None

    # Extract version from file_path_y in final_df
    final_df['version_candidate'] = final_df['file_path_y'].apply(extract_file_path_version)

    # Construct a join key for merging
    final_df['join_key'] = final_df['SF#'].astype(str) + final_df['version_candidate'].astype(str)
    purity_estimates['join_key'] = purity_estimates['SF#'].astype(str) + purity_estimates['version_extracted'].astype(str)

    # Remove specific patients from purity_estimates
    purity_estimates = purity_estimates[~purity_estimates['Patient'].isin(['P516', 'P469'])]

    # Merge the data
    merged_df = pd.merge(
        final_df,
        purity_estimates,
        on='join_key',
        how='left',  # Keep all rows from final_df
        suffixes=('', '_purity')
    )

    # Assign 'plot_purity' based on 'Histology'
    merged_df['plot_purity'] = np.where(
        merged_df['Histology'].isin(['GBM', 'Oligo']),  # Condition
        merged_df['FACET_purity'],                      # Value if condition is True
        merged_df['pyclone.IDH.purity']                 # Value if condition is False (Astro)
    )

    # Remove duplicates
    merged_df = merged_df.drop_duplicates()

    return merged_df


########################################################################
# Sequence Processing & Entropy Calculation
########################################################################

def calculate_shannon_entropy(kmer_counts):
    """
    Calculate Shannon entropy given k-mer counts.
    
    Parameters:
        kmer_counts (dict): A dictionary of k-mer counts.
    
    Returns:
        float: The Shannon entropy.
    """
    total_kmers = sum(kmer_counts.values())
    entropy = -sum((count / total_kmers) * math.log2(count / total_kmers)
                   for count in kmer_counts.values() if count > 0)
    return entropy

def process_sequence_with_entropy(sequence, k=3):
    """
    Clean a nucleotide sequence, generate k-mer counts, and compute its Shannon entropy.
    
    Parameters:
        sequence (str): The nucleotide sequence.
        k (int): The k-mer length (default is 3).
        
    Returns:
        tuple: A tuple (kmer_counts, entropy)
    """
    cleaned_sequence = re.sub(r'[^ATCGN]', '', sequence)
    kmer_counts = Counter([cleaned_sequence[i:i+k] for i in range(len(cleaned_sequence) - k + 1)])
    entropy = calculate_shannon_entropy(kmer_counts)
    return kmer_counts, entropy

def process_sequence(sequence):
    """
    Clean a nucleotide sequence by removing any character that is not A, T, C, G, or N.
    
    Parameters:
        sequence (str): The input nucleotide sequence.
        
    Returns:
        str: The cleaned sequence.
    """
    cleaned_sequence = re.sub(r'[^ATCGN]', '', sequence)
    return cleaned_sequence

########################################################################
# Fusion Coordinates and Genomic Position Parsing
########################################################################

def parse_fusion_coords(fusion_str):
    """
    Parse a fusion coordinate string of the form 'chrX:12345_chrY:67890'
    and return a tuple of (chr1, pos1, chr2, pos2).
    
    Parameters:
        fusion_str (str): The fusion coordinate string.
        
    Returns:
        tuple: (chr1, pos1, chr2, pos2) or (None, None, None, None) if no match.
    """
    pattern = r'(chr[0-9XY]+):(\d+)_(chr[0-9XY]+):(\d+)'
    match = re.search(pattern, fusion_str)
    if not match:
        return (None, None, None, None)
    chr1, pos1, chr2, pos2 = match.groups()
    return chr1, int(pos1), chr2, int(pos2)

def parse_genomic_string(entry):
    """
    Extract chromosome and a representative (midpoint) coordinate from a genomic string.
    
    Parameters:
        entry (str): A string containing genomic coordinates (e.g., "chr7:55249071_chr5:30014310").
    
    Returns:
        tuple: (chrom, midpoint) where midpoint is derived from the first (or averaged)
               coordinate(s).
    """
    matches = re.findall(r'(chr[0-9XY]+):(\d+)', entry)
    if not matches or len(matches) < 1:
        return None, None
    chrom1, pos1 = matches[0]
    pos1 = int(pos1)
    if len(matches) == 1:
        midpoint = pos1
        chrom = chrom1
    else:
        chrom2, pos2 = matches[1]
        pos2 = int(pos2)
        if chrom1 == chrom2:
            midpoint = (pos1 + pos2) // 2
            chrom = chrom1
        else:
            midpoint = pos1
            chrom = chrom1
    return chrom, midpoint

def get_genome_coord(chrom, pos, cum_offset):
    """
    Convert a chromosome coordinate to a linear genome coordinate using provided cumulative offsets.
    
    Parameters:
        chrom (str): Chromosome name (e.g., 'chr7').
        pos (int): Position on the chromosome.
        cum_offset (dict): A dictionary mapping chromosomes to their cumulative offset.
        
    Returns:
        int: The linear genome coordinate.
    """
    return cum_offset.get(chrom, 0) + pos

########################################################################
# Transcript and Exon Extraction (using pyensembl and pyfaidx)
########################################################################

# Assume that a pyensembl data object (e.g. EnsemblRelease) is available as `data`
# and that a FASTA extractor (e.g., an instance of FastaStringExtractor) is available as `fasta_extractor`.

@lru_cache(maxsize=None)
def get_transcript(transcript_id):
    """
    Retrieve transcript information using pyensembl.
    
    Parameters:
        transcript_id (str): The transcript ID.
        
    Returns:
        Transcript: The transcript object.
    """
    transcript_id = transcript_id.split('.')[0]
    return data.transcript_by_id(transcript_id)

def obtain_exons(transcript_id):
    """
    Retrieve a list of exons for a given transcript.
    
    Parameters:
        transcript_id (str): The transcript ID.
    
    Returns:
        list: A list of tuples (chrom, start, end, strand) for each exon.
    """
    transcript = get_transcript(transcript_id)
    exons = transcript.exons
    return [(exon.contig, exon.start, exon.end, exon.strand) for exon in exons]

def get_tss(transcript_id):
    """
    Get the transcription start site (TSS) for a given transcript.
    
    Parameters:
        transcript_id (str): The transcript ID.
    
    Returns:
        int: The TSS coordinate.
    """
    transcript = get_transcript(transcript_id)
    return transcript.start if transcript.strand == '+' else transcript.end

def get_5prime_sequence(transcript_id, breakpoint):
    """
    Retrieve the 5' sequence of a transcript up to a given breakpoint.
    
    Parameters:
        transcript_id (str): The transcript ID.
        breakpoint (int): The coordinate breakpoint.
    
    Returns:
        Seq: The 5' sequence (as a Bio.Seq object) concatenated from relevant exons.
    """
    breakpoint = int(breakpoint)
    sequences = []
    exons = obtain_exons(transcript_id)
    strand = exons[0][3]

    if strand == '+':
        for chrom, start, end, _ in exons:
            if end <= breakpoint:
                seq = fasta_extractor.extract(Interval(f'chr{chrom}', start - 1, end))
                sequences.append(seq)
            else:
                seq = fasta_extractor.extract(Interval(f'chr{chrom}', start - 1, breakpoint))
                sequences.append(seq)
                break
    else:
        for chrom, start, end, _ in reversed(exons):
            if breakpoint >= start:
                seq = fasta_extractor.extract(Interval(f'chr{chrom}', start - 1, end))
                sequences.append(seq)
            else:
                seq = fasta_extractor.extract(Interval(f'chr{chrom}', start - 1, breakpoint))
                sequences.append(seq)
                break
    return Seq(''.join(sequences))

def get_3prime_sequence(transcript_id, breakpoint):
    """
    Retrieve the 3' sequence of a transcript starting from a given breakpoint.
    
    Parameters:
        transcript_id (str): The transcript ID.
        breakpoint (int): The coordinate breakpoint.
    
    Returns:
        Seq: The 3' sequence (as a Bio.Seq object) concatenated from relevant exons.
    """
    breakpoint = int(breakpoint)
    sequences = []
    exons = obtain_exons(transcript_id)
    strand = exons[0][3]

    if strand == '+':
        for chrom, start, end, _ in exons:
            if start >= breakpoint:
                seq = fasta_extractor.extract(Interval(f'chr{chrom}', start - 1, end))
                sequences.append(seq)
            elif end >= breakpoint:
                seq = fasta_extractor.extract(Interval(f'chr{chrom}', breakpoint - 1, end))
                sequences.append(seq)
    else:
        for chrom, start, end, _ in reversed(exons):
            if end <= breakpoint:
                seq = fasta_extractor.extract(Interval(f'chr{chrom}', start - 1, end))
                seq = str(Seq(seq).reverse_complement())
                sequences.append(seq)
            elif start <= breakpoint:
                seq = fasta_extractor.extract(Interval(f'chr{chrom}', start - 1, breakpoint))
                seq = str(Seq(seq).reverse_complement())
                sequences.append(seq)
    return Seq(''.join(sequences))

def get_sequence_or_none(transcript_id, breakpoint):
    """
    Obtain the 5' sequence of a transcript up to a breakpoint.
    Returns None if the breakpoint is not valid relative to the TSS.
    
    Parameters:
        transcript_id (str): The transcript ID.
        breakpoint (int): The breakpoint coordinate.
    
    Returns:
        Seq or None: The sequence if valid, otherwise None.
    """
    sequence = get_5prime_sequence(transcript_id, breakpoint)
    tss = int(get_tss(transcript_id))
    strand = get_transcript(transcript_id).strand
    if (strand == '+' and breakpoint < tss) or (strand == '-' and breakpoint > tss):
        return None
    return sequence

########################################################################
# Protein Sequence & Start Codon Analysis
########################################################################

def find_first_methionine(nucleotide_sequence):
    """
    For a given nucleotide sequence, find the first methionine (start codon)
    in each of the three reading frames.
    
    Parameters:
        nucleotide_sequence (str): The nucleotide sequence.
    
    Returns:
        dict: A dictionary with keys as frame numbers (0, 1, 2) and values as
              a dict with details of the first methionine (nucleotide positions, codon, etc.).
    """
    results = {}
    for frame in range(3):
        trimmed_seq = nucleotide_sequence[frame:]
        codon_seq = trimmed_seq[:((len(trimmed_seq)//3)*3)]
        protein_seq = codon_seq.translate()
        position_in_protein = protein_seq.find('M')
        if position_in_protein != -1:
            nucleotide_start = frame + position_in_protein * 3
            nucleotide_end = nucleotide_start + 3
            results[frame] = {
                'nucleotide_position': (nucleotide_start, nucleotide_end),
                'codon': nucleotide_sequence[nucleotide_start:nucleotide_end],
                'protein_position': position_in_protein,
                'frame': frame
            }
    return results

########################################################################
# Cytoband-based Chromosome Arm Processing
########################################################################

def find_pq_boundaries_acen_midpoint(df):
    """
    Given a cytoband DataFrame, return a dictionary mapping each chromosome to 
    the p-q boundary coordinate based on the midpoint of the centromeric ('acen') region.
    
    Parameters:
        df (pandas.DataFrame): A cytoband DataFrame with columns including 'chrom', 'start', 'end', 'annot', 'gieStain'.
    
    Returns:
        dict: {chromosome: boundary_coord}
    """
    boundaries = {}
    for chrom, grp in df.groupby('chrom'):
        acen_bands = grp[grp['gieStain'] == 'acen']
        if len(acen_bands) > 0:
            start_acen = acen_bands['start'].min()
            end_acen   = acen_bands['end'].max()
            boundary   = (start_acen + end_acen) / 2.0
            boundaries[chrom] = boundary
        else:
            boundaries[chrom] = None
    return boundaries

def label_p_or_q(chrom, coord, boundaries_dict):
    """
    Given a chromosome and a coordinate, return 'p' if the coordinate is below
    the centromere midpoint (p-arm) or 'q' otherwise.
    
    Parameters:
        chrom (str): Chromosome name (e.g., "chr7").
        coord (int): Genomic coordinate.
        boundaries_dict (dict): Dictionary of chromosome boundaries.
    
    Returns:
        str or None: 'p' or 'q' if a boundary is found, else None.
    """
    boundary = boundaries_dict.get(chrom, None)
    if boundary is None:
        return None
    return 'p' if coord < boundary else 'q'
