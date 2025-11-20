# Image Comparison Tool using Vertex AI Multimodal Embeddings + Gemini 2.5 Pro

This script efficiently compares images from two folders to find duplicates using a two-stage approach combining Google's Vertex AI Multimodal Embeddings and Gemini 2.5 Pro. It can identify images as the same even when they have different file names, sizes, or file formats.

## Features

- **Two-stage comparison approach** for efficiency:
  1. Fast multimodal embeddings for all images
  2. Similarity scoring using cosine similarity
  3. Gemini 2.5 Pro verification for top candidates only
- Handles different file formats (JPG, PNG, WEBP, HEIC, etc.)
- Ignores compression artifacts and encoding differences
- Multithreaded processing for optimal performance
- Configurable rate limiting and concurrency
- Detailed progress tracking and logging
- Results saved to both file and console output

## Prerequisites

1. **Google Cloud Project**: You need a Google Cloud project with Vertex AI API enabled
2. **Authentication**: Authenticate with Google Cloud
3. **Python 3.8+**: Python 3.8 or higher installed

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Enable Vertex AI API

Make sure the Vertex AI API is enabled in your Google Cloud project:

```bash
gcloud services enable aiplatform.googleapis.com
```

### 3. Authenticate with Google Cloud

You have several options for authentication:

**Option A: Using gcloud CLI (Recommended for local development)**
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**Option B: Using Service Account**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

**Option C: In Cloud environments (Cloud Shell, Compute Engine, etc.)**
Authentication is handled automatically.

## Usage

### Basic Usage

```bash
python compare_images.py <folder1> <folder2> --project-id YOUR_PROJECT_ID
```

### Example

```bash
python compare_images.py ./images/set1 ./images/set2 --project-id my-gcp-project
```

### Full Options

```bash
python compare_images.py <folder1> <folder2> \
  --project-id YOUR_PROJECT_ID \
  --location us-central1 \
  --log-file output.log \
  --max-workers 10 \
  --rate-limit 600 \
  --similarity-threshold 0.95 \
  --verify-top-n 100
```

### Parameters

- `folder1`: Path to the first folder containing images (required)
- `folder2`: Path to the second folder containing images (required)
- `--project-id`: Your Google Cloud project ID (required)
- `--location`: Google Cloud region (default: us-central1)
- `--log-file`: Path to log file (default: output.log)
- `--max-workers`: Number of concurrent threads (default: 10)
- `--rate-limit`: Maximum API calls per minute (default: 600)
- `--similarity-threshold`: Minimum cosine similarity for candidates (default: 0.95)
- `--verify-top-n`: Number of top candidates to verify with Gemini, 0 for all (default: 100)

Available regions include:
- `us-central1` (default)
- `us-east4`
- `us-west1`
- `europe-west1`
- `asia-northeast1`

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- HEIC (.heic)
- HEIF (.heif)
- BMP (.bmp)
- GIF (.gif)

## How It Works

This script uses a **three-stage approach** for efficient and accurate comparison:

### Stage 1: Generate Embeddings
- Gets multimodal embeddings for all images using `multimodalembedding@001`
- Embeddings are 1408-dimensional vectors representing visual content
- Fast processing with multithreading and rate limiting
- Progress tracking shows images/sec and completion status

### Stage 2: Find Similar Pairs
- Compares all embeddings using cosine similarity
- Only pairs above the similarity threshold (default 0.95) are considered candidates
- Extremely fast comparison (thousands of comparisons per second)
- Candidates are sorted by similarity score

### Stage 3: Verify with Gemini
- Top N candidates (default: 100) are analyzed by Gemini 2.5 Pro
- AI verifies if images are exact matches or just very similar
- Provides detailed analysis of differences (if any)
- Multithreaded processing with rate limiting

### Why This Approach?

- **Efficient**: Only uses expensive Gemini calls on promising candidates
- **Scalable**: Can handle large image collections (hundreds of images)
- **Accurate**: Combines fast similarity scoring with detailed AI verification
- **Cost-effective**: Minimizes API costs by pre-filtering with embeddings

## Output Example

```
Initializing Vertex AI (Project: my-project, Location: us-central1)...

Scanning folder 1: 1st_images
Found 180 images

Scanning folder 2: 2nd_images
Found 242 images

================================================================================
STAGE 1: Getting embeddings for all images
================================================================================

Getting embeddings for 180 images...
  Progress: 180/180 (100.0%) - 12.5 imgs/sec
Completed 180 embeddings in 14.4s

Getting embeddings for 242 images...
  Progress: 242/242 (100.0%) - 13.2 imgs/sec
Completed 242 embeddings in 18.3s

================================================================================
STAGE 2: Finding similar pairs using embeddings
================================================================================

Finding similar pairs (threshold: 0.95)...
  Progress: 43560/43560 (100.0%) - 25000 comparisons/sec - Found 45 candidates
Found 45 candidate pairs in 1.7s

================================================================================
STAGE 3: Verifying candidates with Gemini 2.5 Pro
================================================================================

Analyzing 45 out of 45 candidates...
(Top similarities: ['0.9989', '0.9985', '0.9982', '0.9978', '0.9975'])

[1/45] image1.jpg vs copy1.png (sim: 0.9989) - ✓ EXACT MATCH
    → These are exactly the same image with identical composition, content, and details.
[2/45] photo2.jpg vs photo2_copy.jpg (sim: 0.9985) - ✓ EXACT MATCH
    → These are exactly the same image with identical composition, content, and details.
[3/45] sunset.png vs sunset_edit.jpg (sim: 0.9982) - ≈ VERY SIMILAR
    → Very similar images of the same scene, but the second has warmer color grading.
...

Results saved to comparison.txt

================================================================================
IMAGE COMPARISON RESULTS
================================================================================

## EXACT MATCHES (32 pairs)

Similarity Score: 0.9989
Folder 1: image1.jpg
Folder 2: copy1.png
Analysis: These are exactly the same image with identical composition, content, and details.
...

## VERY SIMILAR (13 pairs)

Similarity Score: 0.9982
Folder 1: sunset.png
Folder 2: sunset_edit.jpg
Analysis: Very similar images of the same scene, but the second has warmer color grading.
...

================================================================================
SUMMARY: 32 exact matches, 13 very similar pairs
================================================================================
```

## Cost Considerations

This script uses two Vertex AI services:

1. **Multimodal Embeddings** (`multimodalembedding@001`)
   - Used for all images in both folders
   - Relatively inexpensive
   - Example: ~422 images = ~422 embedding calls

2. **Gemini 2.5 Pro**
   - Only used for top N candidates (default: 100)
   - More expensive but limited usage
   - Controlled by `--verify-top-n` parameter

For pricing details, see: [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing)

### Cost Optimization Tips

1. **Adjust similarity threshold**: Increase `--similarity-threshold` (e.g., 0.97) to reduce candidates
2. **Limit verification**: Use `--verify-top-n` to verify only top candidates
3. **Use Flash model**: For faster/cheaper verification, modify line 320:
   ```python
   gemini_model = GenerativeModel("gemini-2.5-flash")  # Instead of gemini-2.5-pro
   ```
4. **Adjust rate limit**: Lower `--rate-limit` if you hit quota limits

## Troubleshooting

### Authentication Errors

If you get authentication errors, ensure:
1. You're logged in: `gcloud auth application-default login`
2. Your project is set: `gcloud config set project YOUR_PROJECT_ID`
3. Vertex AI API is enabled

### Permission Errors

Ensure your account/service account has the necessary permissions:
- `roles/aiplatform.user` or `roles/aiplatform.admin`

Grant permissions:
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL" \
  --role="roles/aiplatform.user"
```

### Region Availability

If you get region errors, Gemini 2.5 Pro might not be available in your region. Try:
- `us-central1` (most commonly available)
- Check current availability: [Gemini model regions](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2025 Kazunori Sato
