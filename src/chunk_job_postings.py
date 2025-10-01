#!/usr/bin/env python3

import pandas as pd
import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys
from text_preprocess import TextCleaner, TextChunker
import re
from bs4 import BeautifulSoup
import unicodedata
import string
from typing import Dict, Optional, Union



def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'chunk_jobs_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def process_job_descriptions(df, cleaning_options: Optional[Dict[str, bool]] = None, 
                           chunking_params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Process, clean, and chunk job descriptions from a DataFrame.
    First chunks the job descriptions, then cleans each chunk.
    
    Args:
        df: DataFrame containing job descriptions
        cleaning_options: Dictionary of cleaning options for TextCleaner
        chunking_params: Dictionary of parameters for chunking:
            - min_words: Minimum number of words per chunk (default: 5)
    
    Returns:
        pd.DataFrame: DataFrame with columns 'text' (chunks), 'label' (empty), and 'job_id'
    """
    # Initialize cleaner
    cleaner = TextCleaner(cleaning_options)
    
    # Default chunking parameters
    default_params = {
        'min_words': 5
    }
    chunk_params = {**default_params, **(chunking_params or {})}
    
    # Lists to store all chunks and their corresponding job IDs
    all_chunks = []
    all_job_ids = []
    
    def split_text(text, min_words):
        # Split by one or more newlines
        chunks = re.split(r'[\n]+', text)
        
        # Remove empty chunks and filter out chunks with fewer than min_words
        chunks = [
            chunk.strip() 
            for chunk in chunks 
            if chunk.strip() and len(chunk.strip().split()) >= min_words
        ]
        
        return chunks
    
    for idx, row in df.iterrows():
        job_desc = row['job_description']
        job_id = row['job_id']  # Use the job_id
        
        # First chunk the text
        try:
            # Split the raw text into chunks
            raw_chunks = split_text(job_desc, min_words=chunk_params['min_words'])
            
            # Then clean each chunk
            cleaned_chunks = [cleaner.clean_text(chunk) for chunk in raw_chunks]
            
            # Filter out any chunks that might have become too short after cleaning
            valid_chunks = [
                chunk for chunk in cleaned_chunks 
                if chunk.strip() and len(chunk.strip().split()) >= chunk_params['min_words']
            ]
            
            all_chunks.extend(valid_chunks)
            # Add the job_id for each chunk from this job
            all_job_ids.extend([job_id] * len(valid_chunks))
        except Exception as e:
            print(f"Error processing job posting {job_id}: {e}")
            continue
    
    # Create DataFrame with chunks, empty labels, and job IDs
    return pd.DataFrame({
        'text': all_chunks,
        'label': '',  # Empty string for labels
        'job_id': all_job_ids  # Add job_id column
    })



def main():
    parser = argparse.ArgumentParser(description='Process and chunk job descriptions')
    parser.add_argument('--input', default='../data/botswana_data.pkl', help='Path to input pickle file')
    parser.add_argument('--output', default='../data/botswana_job_postings_chunks.pkl', help='Path to output pickle file')
    parser.add_argument('--log-dir', default='logs', help='Directory for log files')
    parser.add_argument('--sample-size', type=int, help='Number of jobs to sample (optional)')
    parser.add_argument('--min-words', type=int, default=5, help='Minimum number of words per chunk')
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting job processing with args: {args}")

    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_pickle(args.input)
        
        # Sample if requested
        if args.sample_size:
            df = df.sample(n=args.sample_size, random_state=42)
            logger.info(f"Sampled {args.sample_size} jobs")
        
        # Custom options
        custom_options = {
            'remove_html': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_phone_numbers': True,
            'remove_special_chars': True,
            'remove_numbers': False,
            'remove_extra_whitespace': True,
            'remove_punctuation': False,
            'convert_bullets': True,
            'fix_sentence_spacing': True,
            'lowercase': False,
            'normalize_unicode': True
        }

        chunking_params = {
            'min_words': args.min_words
        }
        
        logger.info(f"Using chunking parameters: min_words={args.min_words}")

        # Process jobs
        logger.info("Processing job descriptions")
        result_df = process_job_descriptions(
            df, 
            cleaning_options=custom_options,
            chunking_params=chunking_params
        )
        print(result_df.head())

        # Save results
        logger.info(f"Saving {len(result_df)} chunks to {args.output}")
        result_df.to_pickle(args.output)

        # Log statistics
        logger.info(f"Total chunks: {len(result_df)}")
        logger.info(f"Average chunk length: {result_df['text'].str.len().mean():.0f} characters")
        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()