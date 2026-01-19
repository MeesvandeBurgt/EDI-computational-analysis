"""
Deframe Analysis Pipeline
--------------------------

This script analyzes transcribed speech data from design sessions to identify
critical narrative frames through n-gram extraction and mutual information scoring.

The pipeline:
1. Loads all Excel transcript files from data/input/excel_format/
2. Preprocesses and tokenizes text
3. Extracts n-grams (n=2,3,4) and computes mutual information scores
4. Filters and ranks candidate framing terms
5. Generates visualizations and exports results

Usage from the command line:

    # From the project root:
    python deframe_analysis.py

    # With custom options:
    python deframe_analysis.py --input-dir data/input/excel_format --output-dir data/output/deframe

Dependencies (install once):

    pip install pandas openpyxl matplotlib seaborn numpy
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for the deframe analysis pipeline."""
    # Input/Output paths
    input_dir: Path = Path("data/input/excel_format")
    output_dir: Path = Path("data/output/deframe")
    
    # N-gram parameters
    min_n: int = 2
    max_n: int = 4
    
    # Filtering thresholds
    min_frequency: int = 2  # Minimum count to consider (hapax legomena filtered)
    min_transcript_frequency: int = 2  # Minimum count within a transcript
    mi_percentile: float = 0.90  # Top percentile for MI-based filtering
    
    # Entrainment heuristics
    min_distinct_turns: int = 2  # Minimum distinct turn indices for entrainment
    min_turn_span: int = 5  # Minimum span between first and last occurrence
    
    # Context extraction
    context_window_size: int = 3  # K turns before/after for context
    
    # Common n-grams to filter out (stoplist)
    stoplist: Set[str] = field(default_factory=lambda: {
        "at the same time", "a little bit", "sort of thing", "and stuff like that",
        "and so on", "or something like that", "kind of", "i think", "you know",
        "i mean", "you see", "in other words", "for example", "as well as"
    })
    
    # Thematic keywords for semantic field inspection
    thematic_keywords: Set[str] = field(default_factory=lambda: {
        "power", "normative", "race", "gender", "colonial", "history", "historical",
        "institutional", "assumption", "bias", "structural", "sociohistorical",
        "critical", "narrative", "frame", "framing", "discourse", "ideology"
    })


def load_transcripts(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all Excel transcript files from the input directory.
    
    Returns a dictionary mapping transcript_id (filename stem) to DataFrame.
    Each DataFrame should have columns: timestamp, speaker, utterance.
    """
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    xlsx_files = sorted(input_dir.glob("*.xlsx"))
    if not xlsx_files:
        logger.warning(f"No .xlsx files found in {input_dir}")
        return {}
    
    transcripts = {}
    for path in xlsx_files:
        logger.info(f"Loading {path.name}...")
        df = pd.read_excel(path, engine="openpyxl")
        
        # Validate required columns
        required_cols = {"timestamp", "speaker", "utterance"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns in {path.name}: {missing}")
        
        transcript_id = path.stem
        transcripts[transcript_id] = df
        logger.info(f"  Loaded {len(df)} speech turns")
    
    return transcripts


def normalize_text(text: str) -> str:
    """
    Normalize text for tokenization:
    - Lowercase
    - Replace em-dashes and smart quotes
    - Preserve apostrophes within words
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    # Replace common Unicode punctuation
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    Uses regex to extract alphanumeric sequences with apostrophes.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    
    # Extract words (alphanumeric + apostrophes within words)
    tokens = re.findall(r"[a-zA-Z0-9']+", normalized)
    # Filter out very short tokens except common ones
    filtered = [t for t in tokens if len(t) > 1 or t in {'i', 'a'}]
    return filtered


def preprocess_transcripts(transcripts: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Preprocess transcripts: add tokens, turn indices, and metadata.
    """
    processed = {}
    
    for transcript_id, df in transcripts.items():
        logger.info(f"Preprocessing {transcript_id}...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Add transcript_id
        df['transcript_id'] = transcript_id
        
        # Add turn index (0-based)
        df['turn_index'] = np.arange(len(df))
        
        # Tokenize utterances
        df['tokens'] = df['utterance'].apply(tokenize)
        
        # Keep original utterance for context display
        df['utterance_original'] = df['utterance']
        
        # Count words per turn
        df['word_count'] = df['tokens'].apply(len)
        
        processed[transcript_id] = df
        logger.info(f"  Tokenized {df['word_count'].sum()} total words")
    
    return processed


def extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from a token list.
    Returns a list of n-gram tuples.
    """
    if len(tokens) < n:
        return []
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def count_ngrams(processed_transcripts: Dict[str, pd.DataFrame], config: AnalysisConfig) -> Tuple[Counter, Counter, Counter, Counter, Dict, int, Dict[int, int]]:
    """
    Count unigrams and n-grams globally and per-transcript.
    
    Returns:
        - unigram_counts: Counter of unigrams globally
        - ngram_counts: Counter of (n, ngram_tuple) globally
        - unigram_counts_by_transcript: Counter[(transcript_id, word)] -> count
        - ngram_counts_by_transcript: Counter[(transcript_id, n, ngram_tuple)] -> count
        - ngram_metadata: Dict[(transcript_id, n, ngram_tuple)] -> {
            'speakers': set of speakers,
            'turn_indices': list of turn indices,
            'occurrences': list of (turn_index, speaker) tuples
          }
        - total_tokens: int
        - total_ngram_positions: Dict[n -> count]
    """
    logger.info("Counting n-grams...")
    
    unigram_counts = Counter()
    ngram_counts = Counter()
    unigram_counts_by_transcript = Counter()
    ngram_counts_by_transcript = Counter()
    ngram_metadata = defaultdict(lambda: {
        'speakers': set(),
        'turn_indices': [],
        'occurrences': []
    })
    
    # Count total tokens for probability calculations
    total_tokens = 0
    total_ngram_positions = defaultdict(int)  # n -> count
    
    for transcript_id, df in processed_transcripts.items():
        for _, row in df.iterrows():
            tokens = row['tokens']
            if not tokens:
                continue
            
            speaker = row['speaker']
            turn_index = row['turn_index']
            
            # Count unigrams
            for token in tokens:
                unigram_counts[token] += 1
                unigram_counts_by_transcript[(transcript_id, token)] += 1
                total_tokens += 1
            
            # Count n-grams for each n
            for n in range(config.min_n, config.max_n + 1):
                ngrams = extract_ngrams(tokens, n)
                total_ngram_positions[n] += len(ngrams)
                
                for ngram_tuple in ngrams:
                    key = (n, ngram_tuple)
                    ngram_counts[key] += 1
                    ngram_counts_by_transcript[(transcript_id, n, ngram_tuple)] += 1
                    
                    # Track metadata
                    meta_key = (transcript_id, n, ngram_tuple)
                    ngram_metadata[meta_key]['speakers'].add(speaker)
                    ngram_metadata[meta_key]['turn_indices'].append(turn_index)
                    ngram_metadata[meta_key]['occurrences'].append((turn_index, speaker))
    
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Total unigrams: {len(unigram_counts)}")
    logger.info(f"  Total n-grams: {len(ngram_counts)}")
    
    return (
        unigram_counts,
        ngram_counts,
        unigram_counts_by_transcript,
        ngram_counts_by_transcript,
        ngram_metadata,
        total_tokens,
        total_ngram_positions
    )


def compute_mi_scores(
    unigram_counts: Counter,
    ngram_counts: Counter,
    total_tokens: int,
    total_ngram_positions: Dict[int, int],
    config: AnalysisConfig
) -> Dict[Tuple[int, Tuple[str, ...]], float]:
    """
    Compute mutual information scores for all n-grams.
    
    MI(w1..n) = log2( p(w1..n) / (prod_i p(w_i)) )
    
    Returns a dictionary mapping (n, ngram_tuple) -> MI score.
    """
    logger.info("Computing mutual information scores...")
    
    mi_scores = {}
    
    # Precompute unigram probabilities
    unigram_probs = {word: count / total_tokens for word, count in unigram_counts.items()}
    
    for (n, ngram_tuple), count in ngram_counts.items():
        # Skip hapax legomena
        if count < config.min_frequency:
            continue
        
        # Compute p(w1..n)
        ngram_prob = count / total_ngram_positions[n]
        
        # Compute product of unigram probabilities
        product_probs = 1.0
        all_words_exist = True
        for word in ngram_tuple:
            if word not in unigram_probs:
                all_words_exist = False
                break
            product_probs *= unigram_probs[word]
        
        if not all_words_exist or product_probs == 0:
            continue
        
        # Compute MI
        if ngram_prob > 0 and product_probs > 0:
            mi = math.log2(ngram_prob / product_probs)
            mi_scores[(n, ngram_tuple)] = mi
    
    logger.info(f"  Computed MI scores for {len(mi_scores)} n-grams")
    
    return mi_scores


def create_ngram_dataframe(
    processed_transcripts: Dict[str, pd.DataFrame],
    ngram_counts_by_transcript: Counter,
    ngram_metadata: Dict,
    mi_scores: Dict[Tuple[int, Tuple[str, ...]], float],
    config: AnalysisConfig
) -> pd.DataFrame:
    """
    Create a comprehensive DataFrame of n-grams with counts, MI scores, and metadata.
    """
    logger.info("Creating n-gram DataFrame...")
    
    records = []
    
    for (transcript_id, n, ngram_tuple), count in ngram_counts_by_transcript.items():
        # Skip if below minimum frequency
        if count < config.min_transcript_frequency:
            continue
        
        # Get global count and MI
        global_key = (n, ngram_tuple)
        # Sum across all transcripts for global count
        global_count = sum(
            ngram_counts_by_transcript.get((tid, n, ngram_tuple), 0)
            for tid in processed_transcripts.keys()
        )
        
        mi_score = mi_scores.get(global_key, None)
        
        # Get metadata
        meta = ngram_metadata.get((transcript_id, n, ngram_tuple), {
            'speakers': set(),
            'turn_indices': [],
            'occurrences': []
        })
        
        turn_indices = sorted(set(meta['turn_indices']))
        speakers = list(meta['speakers'])
        
        # Compute entrainment flags
        num_distinct_turns = len(turn_indices)
        turn_span = max(turn_indices) - min(turn_indices) if turn_indices else 0
        num_speakers = len(speakers)
        
        entrained_across_turns = (
            num_distinct_turns >= config.min_distinct_turns and
            turn_span >= config.min_turn_span
        )
        entrained_across_speakers = num_speakers >= 2
        
        # Convert ngram tuple to string
        ngram_string = ' '.join(ngram_tuple)
        
        # Check if in stoplist
        in_stoplist = ngram_string in config.stoplist
        
        records.append({
            'transcript_id': transcript_id,
            'n': n,
            'ngram': ngram_string,
            'count_in_transcript': count,
            'global_count': global_count,
            'global_MI': mi_score,
            'num_speakers': num_speakers,
            'num_distinct_turns': num_distinct_turns,
            'first_turn_index': min(turn_indices) if turn_indices else None,
            'last_turn_index': max(turn_indices) if turn_indices else None,
            'turn_span': turn_span,
            'entrained_across_turns': entrained_across_turns,
            'entrained_across_speakers': entrained_across_speakers,
            'in_stoplist': in_stoplist,
            'speakers': ', '.join(sorted(speakers))
        })
    
    df = pd.DataFrame(records)
    logger.info(f"  Created DataFrame with {len(df)} n-gram records")
    
    return df


def filter_candidates(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """
    Apply filtering to identify candidate framing terms.
    """
    logger.info("Filtering candidates...")
    
    # Start with all n-grams
    candidates = df.copy()
    
    # Step 1: Remove hapax legomena (already done in counting, but double-check)
    candidates = candidates[candidates['global_count'] >= config.min_frequency]
    
    # Step 2: Remove stoplist items
    candidates = candidates[~candidates['in_stoplist']]
    
    # Step 3: Filter by MI percentile (separately for each n)
    candidates_filtered = []
    for n in range(config.min_n, config.max_n + 1):
        n_subset = candidates[candidates['n'] == n].copy()
        if len(n_subset) > 0 and n_subset['global_MI'].notna().any():
            mi_threshold = n_subset['global_MI'].quantile(config.mi_percentile)
            n_subset = n_subset[n_subset['global_MI'] >= mi_threshold]
            candidates_filtered.append(n_subset)
    
    if candidates_filtered:
        candidates = pd.concat(candidates_filtered, ignore_index=True)
    
    # Step 4: Sort by MI score (descending)
    candidates = candidates.sort_values('global_MI', ascending=False, na_position='last')
    
    logger.info(f"  {len(candidates)} candidates after filtering")
    
    return candidates


def extract_context(
    transcript_id: str,
    ngram: str,
    turn_index: int,
    processed_transcripts: Dict[str, pd.DataFrame],
    config: AnalysisConfig
) -> pd.DataFrame:
    """
    Extract context window around an n-gram occurrence.
    """
    df = processed_transcripts[transcript_id]
    
    start_idx = max(0, turn_index - config.context_window_size)
    end_idx = min(len(df), turn_index + config.context_window_size + 1)
    
    context = df.iloc[start_idx:end_idx].copy()
    context['is_occurrence'] = context['turn_index'] == turn_index
    
    return context


def inspect_ngram(
    transcript_id: str,
    ngram: str,
    candidates_df: pd.DataFrame,
    processed_transcripts: Dict[str, pd.DataFrame],
    config: AnalysisConfig
) -> None:
    """
    Print detailed information about an n-gram for manual inspection.
    """
    ngram_data = candidates_df[
        (candidates_df['transcript_id'] == transcript_id) &
        (candidates_df['ngram'] == ngram)
    ]
    
    if ngram_data.empty:
        print(f"N-gram '{ngram}' not found in transcript '{transcript_id}'")
        return
    
    row = ngram_data.iloc[0]
    
    print("=" * 80)
    print(f"N-gram: {row['ngram']}")
    print(f"Transcript: {transcript_id}")
    print(f"N: {row['n']}")
    print(f"Count in transcript: {row['count_in_transcript']}")
    print(f"Global count: {row['global_count']}")
    print(f"MI score: {row['global_MI']:.3f}")
    print(f"Speakers: {row['speakers']}")
    print(f"Distinct turns: {row['num_distinct_turns']}")
    print(f"Turn span: {row['turn_span']}")
    print(f"Entrained across turns: {row['entrained_across_turns']}")
    print(f"Entrained across speakers: {row['entrained_across_speakers']}")
    print("=" * 80)
    
    # Get all occurrences
    df = processed_transcripts[transcript_id]
    ngram_lower = ngram.lower()
    
    print("\nOccurrences in context:")
    print("-" * 80)
    
    for idx, row_df in df.iterrows():
        utterance_lower = normalize_text(str(row_df['utterance']))
        if ngram_lower in utterance_lower:
            turn_idx = row_df['turn_index']
            context = extract_context(transcript_id, ngram, turn_idx, processed_transcripts, config)
            
            print(f"\nTurn {turn_idx} ({row_df['speaker']}):")
            for _, ctx_row in context.iterrows():
                marker = ">>> " if ctx_row['is_occurrence'] else "    "
                print(f"{marker}Turn {ctx_row['turn_index']} ({ctx_row['speaker']}): {ctx_row['utterance_original']}")
            print("-" * 80)


def create_visualizations(
    candidates_df: pd.DataFrame,
    processed_transcripts: Dict[str, pd.DataFrame],
    config: AnalysisConfig
) -> None:
    """
    Create visualizations: MI vs frequency scatterplots and n-gram timelines.
    """
    logger.info("Creating visualizations...")
    
    if candidates_df.empty:
        logger.warning("No candidates to visualize. Skipping visualization step.")
        return
    
    plots_dir = config.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style('ticks')
    sns.set_context('paper', font_scale=1.2)
    
    # 1. MI vs Frequency scatterplots (one per n)
    for n in range(config.min_n, config.max_n + 1):
        n_data = candidates_df[candidates_df['n'] == n].copy()
        if n_data.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by entrainment
        colors = n_data.apply(
            lambda row: 'red' if row['entrained_across_turns'] else 'blue',
            axis=1
        )
        
        scatter = ax.scatter(
            n_data['global_count'],
            n_data['global_MI'],
            c=colors,
            alpha=0.6,
            s=50
        )
        
        ax.set_xlabel('Global Count (log scale)', fontsize=12)
        ax.set_ylabel('Mutual Information Score', fontsize=12)
        ax.set_title(f'MI vs Frequency: {n}-grams', fontsize=14)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Entrained across turns'),
            Patch(facecolor='blue', label='Not entrained')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"mi_vs_frequency_{n}grams.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Timeline plots for top candidates per transcript
    for transcript_id in processed_transcripts.keys():
        transcript_candidates = candidates_df[
            (candidates_df['transcript_id'] == transcript_id) &
            (candidates_df['entrained_across_turns'])
        ].head(10)  # Top 10 by MI
        
        if transcript_candidates.empty:
            continue
        
        df = processed_transcripts[transcript_id]
        num_turns = len(df)
        
        fig, axes = plt.subplots(
            len(transcript_candidates),
            1,
            figsize=(max(10, num_turns / 10), len(transcript_candidates) * 1.5),
            sharex=True
        )
        
        if len(transcript_candidates) == 1:
            axes = [axes]
        
        for idx, (_, candidate_row) in enumerate(transcript_candidates.iterrows()):
            ax = axes[idx]
            ngram = candidate_row['ngram']
            
            # Find all occurrences
            occurrences = []
            for _, row_df in df.iterrows():
                utterance_lower = normalize_text(str(row_df['utterance']))
                if ngram.lower() in utterance_lower:
                    occurrences.append(row_df['turn_index'])
            
            # Plot timeline
            if occurrences:
                ax.vlines(occurrences, 0, 1, colors='red', linewidths=2, label=ngram)
                ax.set_ylim(0, 1.2)
                ax.set_yticks([])
                ax.set_ylabel(ngram[:30] + '...' if len(ngram) > 30 else ngram, rotation=0, ha='right', va='center')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3, axis='x')
        
        axes[-1].set_xlabel('Turn Index', fontsize=12)
        plt.suptitle(f'Timeline of Top Candidate N-grams: {transcript_id}', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(plots_dir / f"timeline_{transcript_id}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"  Saved visualizations to {plots_dir}")


def export_results(
    candidates_df: pd.DataFrame,
    processed_transcripts: Dict[str, pd.DataFrame],
    config: AnalysisConfig
) -> None:
    """
    Export results to CSV files in the candidates/ subdirectory.
    """
    logger.info("Exporting results...")
    
    # Create candidates directory
    candidates_dir = config.output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    
    # Per-transcript candidates
    for transcript_id in processed_transcripts.keys():
        transcript_candidates = candidates_df[
            candidates_df['transcript_id'] == transcript_id
        ].copy()
        
        if not transcript_candidates.empty:
            # Drop complex columns for CSV
            export_cols = [
                'transcript_id', 'n', 'ngram', 'count_in_transcript', 'global_count',
                'global_MI', 'num_speakers', 'num_distinct_turns', 'turn_span',
                'entrained_across_turns'
            ]
            transcript_candidates[export_cols].to_csv(
                candidates_dir / f"{transcript_id}_candidates.csv",
                index=False
            )
    
    # Global summary
    global_summary = candidates_df.groupby(['n', 'ngram']).agg({
        'global_count': 'first',
        'global_MI': 'first',
        'transcript_id': lambda x: ', '.join(sorted(set(x))),
        'count_in_transcript': 'sum'
    }).reset_index()
    
    global_summary = global_summary.sort_values('global_MI', ascending=False, na_position='last')
    global_summary.to_csv(
        candidates_dir / "all_candidates_summary.csv",
        index=False
    )
    
    logger.info(f"  Exported CSV results to {candidates_dir}")


def run_analysis(config: AnalysisConfig) -> None:
    """
    Run the complete analysis pipeline.
    """
    logger.info("=" * 80)
    logger.info("Starting Deframe Analysis Pipeline")
    logger.info("=" * 80)
    
    # Step 1: Load transcripts
    transcripts = load_transcripts(config.input_dir)
    if not transcripts:
        logger.error("No transcripts found. Exiting.")
        return
    
    # Step 2: Preprocess
    processed_transcripts = preprocess_transcripts(transcripts)
    
    # Step 3: Count n-grams
    (
        unigram_counts,
        ngram_counts,
        unigram_counts_by_transcript,
        ngram_counts_by_transcript,
        ngram_metadata,
        total_tokens,
        total_ngram_positions
    ) = count_ngrams(processed_transcripts, config)
    
    # Step 4: Compute MI scores
    mi_scores = compute_mi_scores(
        unigram_counts,
        ngram_counts,
        total_tokens,
        total_ngram_positions,
        config
    )
    
    # Step 5: Create n-gram DataFrame
    ngram_df = create_ngram_dataframe(
        processed_transcripts,
        ngram_counts_by_transcript,
        ngram_metadata,
        mi_scores,
        config
    )
    
    # Step 6: Filter candidates
    candidates_df = filter_candidates(ngram_df, config)
    
    # Step 7: Export results
    export_results(candidates_df, processed_transcripts, config)
    
    # Step 8: Create visualizations
    create_visualizations(candidates_df, processed_transcripts, config)
    
    logger.info("=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze transcripts for critical narrative frames using n-gram MI scoring"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/input/excel_format"),
        help="Directory containing Excel transcript files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/deframe"),
        help="Directory for output files"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum n-gram frequency to consider"
    )
    parser.add_argument(
        "--mi-percentile",
        type=float,
        default=0.90,
        help="MI percentile threshold (0-1)"
    )
    
    args = parser.parse_args()
    
    config = AnalysisConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_frequency=args.min_frequency,
        mi_percentile=args.mi_percentile
    )
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the complete analysis pipeline
    run_analysis(config)

