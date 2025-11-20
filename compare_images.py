#!/usr/bin/env python3
"""
Image Comparison Script using Vertex AI Multimodal Embeddings + Gemini

This script compares images from two folders using a two-stage approach:
1. Get multimodal embeddings for all images (fast)
2. Find similar pairs using dot product similarity
3. Verify top candidates with Gemini 2.5 Pro (only when needed)
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np

import vertexai
from vertexai.vision_models import Image as VisionImage, MultiModalEmbeddingModel
from vertexai.generative_models import GenerativeModel, Part, Image


# Supported image formats
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.bmp', '.gif'}

# Log file
LOG_FILE = None
LOG_LOCK = threading.Lock()


def log_print(message, end='\n'):
    """Print to console and write to log file (thread-safe)."""
    with LOG_LOCK:
        print(message, end=end)
        sys.stdout.flush()
        if LOG_FILE:
            LOG_FILE.write(message + end)
            LOG_FILE.flush()


class RateLimiter:
    """Rate limiter to control API calls per minute."""

    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_call = time.time()


def get_image_files(folder_path: str) -> List[str]:
    """Get all image files from a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        log_print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)

    if not folder.is_dir():
        log_print(f"Error: '{folder_path}' is not a directory.")
        sys.exit(1)

    image_files = []
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(str(file_path))

    return sorted(image_files)


def get_embeddings_batch(model: MultiModalEmbeddingModel, image_paths: List[str],
                         rate_limiter: RateLimiter, max_workers: int = 10) -> Dict[str, np.ndarray]:
    """
    Get embeddings for a batch of images using multithreading.

    Returns: Dict mapping image path to embedding vector
    """
    embeddings = {}
    embeddings_lock = threading.Lock()

    def get_single_embedding(image_path: str):
        """Get embedding for a single image."""
        try:
            rate_limiter.wait()

            # Load image
            image = VisionImage.load_from_file(image_path)

            # Get embedding
            result = model.get_embeddings(image=image, dimension=1408)
            embedding = np.array(result.image_embedding)

            with embeddings_lock:
                embeddings[image_path] = embedding

            return True

        except Exception as e:
            log_print(f"Error getting embedding for {Path(image_path).name}: {e}")
            return False

    # Process embeddings with multithreading
    log_print(f"Getting embeddings for {len(image_paths)} images...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_single_embedding, path) for path in image_paths]

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == len(image_paths):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                log_print(f"  Progress: {completed}/{len(image_paths)} ({completed/len(image_paths)*100:.1f}%) - {rate:.1f} imgs/sec")

    elapsed = time.time() - start_time
    log_print(f"Completed {len(embeddings)} embeddings in {elapsed:.1f}s")

    return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


def find_similar_pairs(embeddings1: Dict[str, np.ndarray],
                       embeddings2: Dict[str, np.ndarray],
                       similarity_threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """
    Find image pairs with high similarity using dot product.

    Returns: List of (image1_path, image2_path, similarity_score) sorted by score
    """
    log_print(f"\nFinding similar pairs (threshold: {similarity_threshold})...")

    candidates = []
    total_comparisons = len(embeddings1) * len(embeddings2)
    current = 0
    start_time = time.time()

    for img1_path, emb1 in embeddings1.items():
        for img2_path, emb2 in embeddings2.items():
            current += 1
            similarity = cosine_similarity(emb1, emb2)

            if similarity >= similarity_threshold:
                candidates.append((img1_path, img2_path, similarity))

            if current % 10000 == 0:
                elapsed = time.time() - start_time
                rate = current / elapsed if elapsed > 0 else 0
                log_print(f"  Progress: {current}/{total_comparisons} ({current/total_comparisons*100:.1f}%) - {rate:.0f} comparisons/sec - Found {len(candidates)} candidates")

    # Sort by similarity (highest first)
    candidates.sort(key=lambda x: x[2], reverse=True)

    elapsed = time.time() - start_time
    log_print(f"Found {len(candidates)} candidate pairs in {elapsed:.1f}s")

    return candidates


def analyze_with_gemini(model: GenerativeModel, img1_path: str, img2_path: str,
                        rate_limiter: RateLimiter) -> Tuple[bool, str]:
    """
    Analyze two images using Gemini 2.5 Pro.

    Returns: (is_exact_match, analysis_text)
    """
    try:
        rate_limiter.wait()

        # Load images
        image1 = Image.load_from_file(img1_path)
        image2 = Image.load_from_file(img2_path)

        prompt = """Compare these two images carefully and provide a detailed analysis.

First, determine if they are EXACTLY the same image (ignoring only minor compression artifacts or file format differences).

Then, describe the comparison in 1-2 sentences, focusing on:
- If exact match: confirm they are identical
- If very similar: what are the key differences? (e.g., different cropping, color grading, resolution, slight variations in composition, etc.)
- If different: briefly state what's different

Format your response as:
MATCH: [YES or NO]
ANALYSIS: [Your 1-2 sentence analysis]

Example responses:
"MATCH: YES
ANALYSIS: These are exactly the same image with identical composition, content, and details."

"MATCH: NO
ANALYSIS: Very similar images of the same scene, but the second image has warmer color grading and slightly different cropping on the right side."

"MATCH: NO
ANALYSIS: Different images showing different subjects/scenes."""

        # Generate response
        response = model.generate_content([prompt, Part.from_image(image1), Part.from_image(image2)])
        answer = response.text.strip()

        # Parse response
        is_match = False
        analysis = answer

        if "MATCH:" in answer:
            lines = answer.split('\n')
            for line in lines:
                if line.startswith("MATCH:"):
                    is_match = "YES" in line.upper()
                elif line.startswith("ANALYSIS:"):
                    analysis = line.replace("ANALYSIS:", "").strip()
                    break
        else:
            # Fallback parsing
            is_match = "YES" in answer.upper() and "MATCH" in answer.upper()

        return is_match, analysis

    except Exception as e:
        log_print(f"Error analyzing {Path(img1_path).name} vs {Path(img2_path).name}: {e}")
        return False, f"Error: {str(e)}"


def format_results(exact_matches: List[Tuple[str, str, float, str]],
                   very_similar: List[Tuple[str, str, float, str]]) -> List[str]:
    """
    Format comparison results as a list of strings.

    Returns: List of formatted result lines
    """
    lines = []
    lines.append("=" * 80)
    lines.append("IMAGE COMPARISON RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Exact matches
    if exact_matches:
        lines.append(f"## EXACT MATCHES ({len(exact_matches)} pairs)")
        lines.append("")
        for img1, img2, similarity, analysis in exact_matches:
            lines.append(f"Similarity Score: {similarity:.4f}")
            lines.append(f"Folder 1: {Path(img1).name}")
            lines.append(f"Folder 2: {Path(img2).name}")
            lines.append(f"Analysis: {analysis}")
            lines.append("")
            lines.append("-" * 80)
            lines.append("")
    else:
        lines.append("## EXACT MATCHES: None found")
        lines.append("")

    # Very similar (but not exact)
    if very_similar:
        lines.append(f"## VERY SIMILAR ({len(very_similar)} pairs)")
        lines.append("")
        for img1, img2, similarity, analysis in very_similar:
            lines.append(f"Similarity Score: {similarity:.4f}")
            lines.append(f"Folder 1: {Path(img1).name}")
            lines.append(f"Folder 2: {Path(img2).name}")
            lines.append(f"Analysis: {analysis}")
            lines.append("")
            lines.append("-" * 80)
            lines.append("")
    else:
        lines.append("## VERY SIMILAR: None found")
        lines.append("")

    # Summary
    lines.append("=" * 80)
    lines.append(f"SUMMARY: {len(exact_matches)} exact matches, {len(very_similar)} very similar pairs")
    lines.append("=" * 80)

    return lines


def find_duplicate_images(folder1: str, folder2: str, project_id: str,
                         location: str = "us-central1",
                         max_workers: int = 10,
                         rate_limit: int = 600,
                         similarity_threshold: float = 0.95,
                         verify_top_n: int = 100):
    """
    Find duplicate images using embeddings + Gemini verification.

    Args:
        folder1: Path to first folder
        folder2: Path to second folder
        project_id: Google Cloud project ID
        location: Google Cloud region
        max_workers: Number of concurrent threads
        rate_limit: Maximum API calls per minute
        similarity_threshold: Minimum cosine similarity to consider as candidate
        verify_top_n: Number of top candidates to verify with Gemini (0 = all candidates)
    """
    # Initialize Vertex AI
    log_print(f"Initializing Vertex AI (Project: {project_id}, Location: {location})...")
    vertexai.init(project=project_id, location=location)

    # Initialize models
    embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    gemini_model = GenerativeModel("gemini-2.5-pro")

    # Rate limiters
    rate_limiter = RateLimiter(rate_limit)

    # Get image files
    log_print(f"\nScanning folder 1: {folder1}")
    images1 = get_image_files(folder1)
    log_print(f"Found {len(images1)} images")

    log_print(f"\nScanning folder 2: {folder2}")
    images2 = get_image_files(folder2)
    log_print(f"Found {len(images2)} images")

    if not images1 or not images2:
        log_print("No images found in one or both folders. Exiting.")
        return

    # Stage 1: Get embeddings for all images
    log_print(f"\n{'='*80}")
    log_print("STAGE 1: Getting embeddings for all images")
    log_print(f"{'='*80}\n")

    embeddings1 = get_embeddings_batch(embedding_model, images1, rate_limiter, max_workers)
    embeddings2 = get_embeddings_batch(embedding_model, images2, rate_limiter, max_workers)

    # Stage 2: Find similar pairs using cosine similarity
    log_print(f"\n{'='*80}")
    log_print("STAGE 2: Finding similar pairs using embeddings")
    log_print(f"{'='*80}\n")

    candidates = find_similar_pairs(embeddings1, embeddings2, similarity_threshold)

    if not candidates:
        log_print(f"\nNo candidates found with similarity >= {similarity_threshold}")
        log_print("\n" + "="*80)
        log_print("RESULTS: No matching images found.")
        log_print("="*80)
        return

    # Stage 3: Verify top candidates with Gemini
    log_print(f"\n{'='*80}")
    log_print("STAGE 3: Verifying candidates with Gemini 2.5 Pro")
    log_print(f"{'='*80}\n")

    # Limit verification if requested
    candidates_to_verify = candidates[:verify_top_n] if verify_top_n > 0 else candidates

    log_print(f"Analyzing {len(candidates_to_verify)} out of {len(candidates)} candidates...")
    log_print(f"(Top similarities: {[f'{c[2]:.4f}' for c in candidates[:min(5, len(candidates))]]})  \n")

    # Analyze candidates with multithreading
    exact_matches = []
    very_similar = []
    results_lock = threading.Lock()
    completed_count = [0]
    completed_lock = threading.Lock()

    def analyze_candidate(candidate):
        """Analyze a single candidate pair."""
        img1, img2, similarity = candidate
        img1_name = Path(img1).name
        img2_name = Path(img2).name

        is_match, analysis = analyze_with_gemini(gemini_model, img1, img2, rate_limiter)

        with completed_lock:
            completed_count[0] += 1
            current = completed_count[0]

        result_type = "✓ EXACT MATCH" if is_match else "≈ VERY SIMILAR"
        log_print(f"[{current}/{len(candidates_to_verify)}] {img1_name} vs {img2_name} (sim: {similarity:.4f}) - {result_type}")
        log_print(f"    → {analysis}")

        with results_lock:
            if is_match:
                exact_matches.append((img1, img2, similarity, analysis))
            else:
                very_similar.append((img1, img2, similarity, analysis))

    # Process with multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_candidate, cand) for cand in candidates_to_verify]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log_print(f"Error in analysis task: {e}")

    # Sort by similarity (highest first)
    exact_matches.sort(key=lambda x: x[2], reverse=True)
    very_similar.sort(key=lambda x: x[2], reverse=True)

    # Format results using helper function
    result_lines = format_results(exact_matches, very_similar)

    # Write results to comparison.txt
    comparison_file = "comparison.txt"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_lines) + '\n')

    log_print(f"\nResults saved to {comparison_file}")

    # Print results to log
    log_print("")
    for line in result_lines:
        log_print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Compare images using Vertex AI Embeddings + Gemini 2.5 Pro"
    )
    parser.add_argument(
        "folder1",
        help="Path to the first folder containing images"
    )
    parser.add_argument(
        "folder2",
        help="Path to the second folder containing images"
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Google Cloud project ID"
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Google Cloud region (default: us-central1)"
    )
    parser.add_argument(
        "--log-file",
        default="output.log",
        help="Path to log file (default: output.log)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of concurrent threads (default: 10)"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=600,
        help="Maximum API calls per minute (default: 600)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Minimum cosine similarity for candidates (default: 0.95)"
    )
    parser.add_argument(
        "--verify-top-n",
        type=int,
        default=100,
        help="Number of top candidates to verify with Gemini, 0 for all (default: 100)"
    )

    args = parser.parse_args()

    # Open log file
    global LOG_FILE
    LOG_FILE = open(args.log_file, 'w', encoding='utf-8')

    try:
        find_duplicate_images(
            folder1=args.folder1,
            folder2=args.folder2,
            project_id=args.project_id,
            location=args.location,
            max_workers=args.max_workers,
            rate_limit=args.rate_limit,
            similarity_threshold=args.similarity_threshold,
            verify_top_n=args.verify_top_n
        )
    finally:
        if LOG_FILE:
            LOG_FILE.close()


if __name__ == "__main__":
    main()
