"""
Triplet Generator using OpenAI API.

Generates subject-relation-object triplets for post-2020 developments
leveraging Pythia's 2020 cutoff for factual memory diffing research.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

import openai
from openai import OpenAI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """Represents a subject-relation-object triplet."""

    subject: str
    relation: str
    object: str
    event_year: Optional[int] = None
    domain: Optional[str] = None
    source: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert triplet to dictionary."""
        return asdict(self)


class TripletResponse(BaseModel):
    """Pydantic model for structured triplet response from OpenAI API."""

    subject: str
    relation: str
    object: str
    event_year: Optional[int] = None
    domain: Optional[str] = None


class TripletsResponse(BaseModel):
    """Pydantic model for structured triplets response array."""

    triplets: list[TripletResponse]


class TripletGenerator:
    """Generate knowledge triplets using OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_retries: int = 3,
        before_year: Optional[int] = None,
        after_year: Optional[int] = None,
        use_web: bool = False,
    ):
        """
        Initialize the triplet generator.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for generation
            temperature: Temperature for generation
            max_retries: Maximum retries for API calls
            before_year: Optional upper bound year for events (inclusive)
            after_year: Optional lower bound year for events (exclusive, e.g., year > after_year)
            use_web: Whether to use OpenAI's web search capability
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.before_year = before_year
        self.after_year = after_year
        self.use_web = use_web

    def _create_system_prompt(self) -> str:
        """Create system prompt for triplet generation."""
        year_constraint = ""
        if self.after_year and self.before_year:
            year_constraint = f"\n1. All facts must occur between {self.after_year} (exclusive) and {self.before_year}"
        elif self.after_year:
            year_constraint = (
                f"\n1. All facts must occur after {self.after_year} (exclusive)"
            )
        elif self.before_year:
            year_constraint = f"\n1. All facts must occur before {self.before_year}"

        web_note = ""
        if self.use_web:
            web_note = "\nYou have access to web search. Use it to verify facts and find recent information."

        return f"""You are an expert knowledge extraction assistant specializing in factual information.
Your task is to generate high-quality subject-relation-object (SRO) triplets representing factual claims with associated event years and domains.

Requirements:{year_constraint}
2. Triplets should be specific, factual, and verifiable
3. Relations should be concise and clear (e.g., "discovered", "won", "released", "invented")
4. Subjects should be people and objects should be specific ideas
5. Include the event year for each triplet when known
6. Include the domain/category for each triplet (e.g., politics, science, technology, sports, business, culture, entertainment)
7. Select especially noteworthy and relevant events{web_note}

You will respond with a JSON object containing an array of triplets."""

    def _create_user_prompt(
        self, num_triplets: int = 10, domain: Optional[str] = None
    ) -> str:
        """Create user prompt for triplet generation."""
        domain_spec = f" Focus on the {domain} domain." if domain else ""
        year_spec = ""
        if self.after_year and self.before_year:
            year_spec = f" Events must be after {self.after_year} (exclusive) and before {self.before_year}."
        elif self.after_year:
            year_spec = f" Events must be after {self.after_year} (exclusive, e.g., year > {self.after_year})."
        elif self.before_year:
            year_spec = f" Events must be before {self.before_year}."

        return f"""Generate {num_triplets} unique subject-relation-object triplets.{domain_spec}{year_spec}
Ensure diversity in topics and relations. Include event_year and domain for each triplet. Return ONLY the JSON array."""

    def generate_triplets(
        self,
        num_triplets: int = 10,
        domain: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> list[Triplet]:
        """
        Generate triplets using OpenAI API with structured outputs.

        Args:
            num_triplets: Number of triplets to generate
            domain: Optional domain focus (e.g., "AI", "science", "politics")
            temperature: Optional temperature override

        Returns:
            List of Triplet objects
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Generating {num_triplets} triplets (attempt {attempt + 1}/{self.max_retries})"
                )

                # Build request parameters with structured output
                request_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self._create_system_prompt()},
                        {
                            "role": "user",
                            "content": self._create_user_prompt(num_triplets, domain),
                        },
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "TripletsResponse",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "triplets": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "subject": {"type": "string"},
                                                "relation": {"type": "string"},
                                                "object": {"type": "string"},
                                                "event_year": {
                                                    "type": ["integer", "null"]
                                                },
                                                "domain": {"type": ["string", "null"]},
                                            },
                                            "required": [
                                                "subject",
                                                "relation",
                                                "object",
                                            ],
                                        },
                                    }
                                },
                                "required": ["triplets"],
                            },
                        },
                    },
                }

                response = self.client.chat.completions.create(**request_params)

                # Parse structured response
                content = response.choices[0].message.content
                response_data = json.loads(content)
                triplet_dicts = response_data.get("triplets", [])

                # Convert to Triplet objects
                triplets = []
                for triplet_dict in triplet_dicts:
                    triplet = Triplet(
                        subject=triplet_dict.get("subject", ""),
                        relation=triplet_dict.get("relation", ""),
                        object=triplet_dict.get("object", ""),
                        event_year=triplet_dict.get("event_year"),
                        domain=triplet_dict.get("domain") or domain,
                        source="openai_gpt",
                        timestamp=datetime.now().isoformat(),
                    )
                    triplets.append(triplet)

                logger.info(f"Successfully generated {len(triplets)} triplets")
                return triplets

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                if attempt == self.max_retries - 1:
                    raise
            except openai.APIError as e:
                logger.warning(f"API error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise

        return []

    def generate_batch(
        self,
        num_triplets: int = 100,
        batch_size: int = 10,
        domains: Optional[list[str]] = None,
    ) -> list[Triplet]:
        """
        Generate a batch of triplets across multiple domains.

        Generates triplets per domain with the specified batch size.

        Args:
            num_triplets: Total number of triplets to generate
            batch_size: Size of each API call per domain
            domains: List of domains to focus on

        Returns:
            List of all generated Triplet objects
        """
        all_triplets = []
        domains = domains or ["general", "science", "technology", "politics", "culture"]

        # Calculate triplets per domain
        triplets_per_domain = num_triplets // len(domains)
        remainder = num_triplets % len(domains)

        for domain_idx, domain in enumerate(domains):
            # Distribute remainder across first few domains
            domain_target = triplets_per_domain + (1 if domain_idx < remainder else 0)

            # Generate triplets for this domain in batches
            domain_triplets = 0
            while domain_triplets < domain_target:
                remaining = domain_target - domain_triplets
                current_batch_size = min(batch_size, remaining)

                triplets = self.generate_triplets(
                    num_triplets=current_batch_size, domain=domain
                )
                all_triplets.extend(triplets)
                domain_triplets += len(triplets)

                logger.info(
                    f"Domain '{domain}': {domain_triplets}/{domain_target} triplets"
                )

        return all_triplets

    def save_triplets(self, triplets: list[Triplet], output_path: str) -> None:
        """
        Save triplets to JSON file.

        Args:
            triplets: List of triplets to save
            output_path: Path to output JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump([t.to_dict() for t in triplets], f, indent=2)

        logger.info(f"Saved {len(triplets)} triplets to {output_path}")

    def load_triplets(self, input_path: str) -> list[Triplet]:
        """
        Load triplets from JSON file.

        Args:
            input_path: Path to input JSON file

        Returns:
            List of Triplet objects
        """
        with open(input_path, "r") as f:
            triplet_dicts = json.load(f)

        triplets = [Triplet(**t) for t in triplet_dicts]
        logger.info(f"Loaded {len(triplets)} triplets from {input_path}")
        return triplets
