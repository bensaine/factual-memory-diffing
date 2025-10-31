#!/usr/bin/env python3
"""
Script to verbalize knowledge triplets into natural language sentences.

Verbalizes triplets using 10 prompt templates and 2 inference templates
(standard and creative). Supports interleaving with background batches.

Usage:
    python data_generation/scripts/verbalize_triplets.py --triplets data/triplets.json --output data/verbalizations.json
    python data_generation/scripts/verbalize_triplets.py --triplets data/triplets.json --inference creative
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verbalization import Verbalizer, InferenceTemplate, PromptTemplateLibrary

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_triplets(triplets_path: str) -> list[dict]:
    """Load triplets from JSON file."""
    with open(triplets_path, "r") as f:
        return json.load(f)


def main():
    """Main entry point for triplet verbalization."""
    parser = argparse.ArgumentParser(
        description="Verbalize knowledge triplets into natural language sentences"
    )
    parser.add_argument(
        "--triplets", type=str, required=True, help="Path to triplets JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/verbalizations.json",
        help="Output JSON file path (default: data/verbalizations.json)",
    )
    parser.add_argument(
        "--inference",
        type=str,
        choices=["standard", "creative"],
        default="standard",
        help="Inference template to use (default: standard)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="OpenAI model to use (default: gpt-4)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--num-exposures",
        type=int,
        default=30,
        help="Number of exposures per fact (default: 30, range 20-40)",
    )
    parser.add_argument(
        "--background-batch-size",
        type=int,
        default=5,
        help="Size of background batches (default: 5)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available prompt templates and exit",
    )

    args = parser.parse_args()

    # List templates if requested
    if args.list_templates:
        logger.info("Available Prompt Templates:")
        logger.info("-" * 80)
        for template in PromptTemplateLibrary.get_templates():
            logger.info(f"  {template.name:20s} - {template.description}")
        return 0

    # Validate num_exposures
    if not (20 <= args.num_exposures <= 40):
        logger.warning(
            f"num_exposures={args.num_exposures} outside recommended range [20, 40]. "
            f"Continuing anyway..."
        )

    logger.info("=" * 80)
    logger.info("TRIPLET VERBALIZATION SCRIPT")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - Triplets file: {args.triplets}")
    logger.info(f"  - Model: {args.model}")
    logger.info(f"  - Temperature: {args.temperature}")
    logger.info(f"  - Inference template: {args.inference}")
    logger.info("  - Prompt templates: 10 (all)")
    logger.info(f"  - Number of exposures per fact: {args.num_exposures}")
    logger.info(f"  - Background batch size: {args.background_batch_size}")
    logger.info(f"  - Output: {args.output}")
    logger.info("=" * 80)

    try:
        # Load triplets
        logger.info(f"Loading triplets from {args.triplets}...")
        triplets = load_triplets(args.triplets)
        logger.info(f"Loaded {len(triplets)} triplets")

        # Initialize verbalizer
        verbalizer = Verbalizer(
            api_key=args.api_key, model=args.model, temperature=args.temperature
        )

        # Determine inference template
        inference_template = (
            InferenceTemplate.CREATIVE
            if args.inference == "creative"
            else InferenceTemplate.STANDARD
        )

        # Verbalize triplets
        logger.info(f"Starting verbalization with {args.inference} inference...")

        # For detailed output, we'll verbalize with interleaving
        results = verbalizer.verbalize_batch(
            triplets=triplets,
            inference_template=inference_template,
            num_exposures=args.num_exposures,
            background_batch_size=args.background_batch_size,
        )

        # Save verbalizations
        logger.info(f"Saving {len(results)} verbalizations to {args.output}...")
        verbalizer.save_verbalizations(results, args.output)

        # Print summary
        logger.info("=" * 80)
        logger.info("VERBALIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total verbalizations: {len(results)}")
        logger.info(f"Saved to: {args.output}")

        # Calculate statistics
        template_counts = {}
        inference_counts = {}
        token_stats = {"min": float("inf"), "max": 0, "total": 0}

        for result in results:
            template_counts[result.prompt_template] = (
                template_counts.get(result.prompt_template, 0) + 1
            )
            inference_counts[result.inference_template] = (
                inference_counts.get(result.inference_template, 0) + 1
            )
            token_stats["total"] += result.token_count
            token_stats["min"] = min(token_stats["min"], result.token_count)
            token_stats["max"] = max(token_stats["max"], result.token_count)

        logger.info("\nStatistics:")
        logger.info(
            f"  Average tokens per sentence: {token_stats['total'] / len(results):.2f}"
        )
        logger.info(
            f"  Min tokens: {token_stats['min']}, Max tokens: {token_stats['max']}"
        )
        logger.info(
            f"  Sentences in target range (5-10 tokens): {sum(1 for r in results if 5 <= r.token_count <= 10)} / {len(results)}"
        )

        logger.info("\nSample verbalizations:")
        for i, result in enumerate(results[:5]):
            logger.info(f"  {i + 1}. [{result.prompt_template}] {result.sentence}")

        return 0

    except KeyboardInterrupt:
        logger.info("Verbalization interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during triplet verbalization: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
