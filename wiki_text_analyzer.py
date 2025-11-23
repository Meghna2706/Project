"""
Wikipedia Text Analysis Tool

This script performs text analysis on Wikipedia articles using Natural Language Processing (NLP)
techniques. It extracts content, removes stopwords, and visualizes word frequency distributions.

Features:
- Web scraping with proper headers
- Text preprocessing and cleaning
- Stopwords removal
- Word frequency analysis
- Data visualization
- Results export
"""

import urllib.request
import argparse
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple

def setup_nltk() -> None:
    """Download required NLTK data."""
    print("Setting up NLTK resources...")
    # Create nltk_data directory if it doesn't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download required NLTK resources
    for resource in ['stopwords', 'punkt', 'averaged_perceptron_tagger']:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=False, raise_on_error=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")
            raise

def get_wikipedia_content(url: str) -> str:
    """
    Fetch content from Wikipedia with proper headers.
    
    Args:
        url (str): Wikipedia article URL
    
    Returns:
        str: Raw HTML content
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                     '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        print(f"Error accessing URL: {e}")
        raise

def process_text(html: str) -> str:
    """
    Extract and clean text from HTML content.
    
    Args:
        html (str): Raw HTML content
    
    Returns:
        str: Cleaned text
    """
    soup = BeautifulSoup(html, 'html5lib')
    # Remove unwanted sections
    for div in soup.find_all(['div', 'table'], class_=['toc', 'thumb', 'infobox']):
        div.decompose()
    return soup.get_text(strip=True)

def analyze_text(text: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Perform text analysis including tokenization and frequency distribution.
    
    Args:
        text (str): Cleaned text
    
    Returns:
        tuple: Clean tokens and frequency distribution
    """
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_tokens = [token.lower() for token in tokens 
                   if token.isalnum() and token.lower() not in stop_words]
    
    return clean_tokens, dict(nltk.FreqDist(clean_tokens))

def visualize_frequencies(freq_dist: Dict[str, int], title: str, output_dir: str) -> None:
    """
    Create and save visualization of word frequencies.
    
    Args:
        freq_dist (dict): Word frequency distribution
        title (str): Plot title
        output_dir (str): Directory to save the plot
    """
    try:
        plt.figure(figsize=(15, 8))
        # Sort words by frequency before plotting
        sorted_items = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)[:30]
        words = [item[0] for item in sorted_items]
        freqs = [item[1] for item in sorted_items]
        
        # Create bar plot
        plt.bar(range(len(words)), freqs, color='skyblue')
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.title(f'Top 30 Words in {title}')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        output_path = os.path.join(output_dir, 'word_frequency.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        plt.close()  # Ensure figure is closed even if save fails

def export_results(freq_dist: Dict[str, int], clean_tokens: List[str], 
                  output_dir: str) -> None:
    """
    Export analysis results to files.
    
    Args:
        freq_dist (dict): Word frequency distribution
        clean_tokens (list): Cleaned tokens
        output_dir (str): Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export frequency distribution to CSV
    df = pd.DataFrame(list(freq_dist.items()), columns=['Word', 'Frequency'])
    df.to_csv(os.path.join(output_dir, 'word_frequencies.csv'), index=False)
    
    # Export summary statistics
    summary = {
        'total_words': len(clean_tokens),
        'unique_words': len(freq_dist),
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'top_10_words': dict(sorted(freq_dist.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:10])
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

def main():
    """Main execution function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Wikipedia Text Analysis Tool')
    parser.add_argument('--url', default='https://en.wikipedia.org/wiki/Indian_Premier_League',
                      help='Wikipedia article URL to analyze')
    parser.add_argument('--output', default='analysis_results',
                      help='Output directory for results')
    args = parser.parse_args()
    
    try:
        # Create output directory first
        os.makedirs(args.output, exist_ok=True)
        
        # Setup
        setup_nltk()
        print(f"Analyzing article: {args.url}")
        
        # Fetch and process content
        html = get_wikipedia_content(args.url)
        text = process_text(html)
        print("Content extracted successfully")
        
        # Analyze text
        clean_tokens, freq_dist = analyze_text(text)
        print(f"Analysis complete. Found {len(clean_tokens)} words, "
              f"{len(freq_dist)} unique words")
        
        # Create visualizations and export results
        visualize_frequencies(freq_dist, 'Wikipedia Article', args.output)
        export_results(freq_dist, clean_tokens, args.output)
        
        print(f"\nResults have been saved to: {os.path.abspath(args.output)}")
        print("Generated files:")
        print("- word_frequency.png (Visualization)")
        print("- word_frequencies.csv (Complete frequency data)")
        print("- analysis_summary.json (Analysis summary)")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()