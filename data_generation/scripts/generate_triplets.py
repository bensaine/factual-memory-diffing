#!/usr/bin/env python3
"""
Script to generate knowledge triplets (subject-relation-object) using OpenAI API.

Usage:
    python data_generation/scripts/generate_triplets.py --num-triplets 100 --output data/triplets.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.triplet_generator import TripletGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for triplet generation."""
    parser = argparse.ArgumentParser(
        description="Generate knowledge triplets for post-2020 developments"
    )
    parser.add_argument(
        "--num-triplets",
        type=int,
        default=50,
        help="Total number of triplets to generate (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size per API call (default: 10)",
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
        "--domains",
        type=str,
        nargs="+",
        default=["general", "science", "technology", "politics", "culture"],
        help="Domains to focus on",
    )
    parser.add_argument(
        "--before-year",
        type=int,
        default=None,
        help="Upper bound year for events (inclusive, e.g., 2020)",
    )
    parser.add_argument(
        "--after-year",
        type=int,
        default=None,
        help="Lower bound year for events (exclusive, e.g., 2020 means year > 2020)",
    )
    parser.add_argument(
        "--use-web",
        action="store_true",
        help="Enable web search for fact verification and recent information",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/triplets.json",
        help="Output JSON file path (default: data/triplets.json)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TRIPLET GENERATION SCRIPT")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - Model: {args.model}")
    logger.info(f"  - Temperature: {args.temperature}")
    logger.info(f"  - Number of triplets: {args.num_triplets}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Domains: {args.domains}")
    logger.info(f"  - Before year: {args.before_year}")
    logger.info(f"  - After year: {args.after_year}")
    logger.info(f"  - Use web search: {args.use_web}")
    logger.info(f"  - Output: {args.output}")
    logger.info("=" * 80)

    try:
        # Initialize generator
        generator = TripletGenerator(
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            before_year=args.before_year,
            after_year=args.after_year,
            use_web=args.use_web,
        )

        # Generate triplets in batches
        logger.info("Starting triplet generation...")
        triplets = generator.generate_batch(
            num_triplets=args.num_triplets,
            batch_size=args.batch_size,
            domains=args.domains,
        )

        # Save triplets
        logger.info(f"Saving {len(triplets)} triplets to {args.output}...")
        generator.save_triplets(triplets, args.output)

        # Print summary
        logger.info("=" * 80)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Generated {len(triplets)} triplets")
        logger.info(f"Saved to: {args.output}")
        logger.info("Sample triplets:")
        for i, triplet in enumerate(triplets[:5]):
            logger.info(
                f"  {i + 1}. {triplet.subject} -> {triplet.relation} -> {triplet.object}"
            )

        return 0

    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during triplet generation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
