"""
Triplet Verbalizer using OpenAI API.

Verbalizes subject-relation-object triplets into natural language sentences
using 10 prompt templates and 2 inference templates.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
from enum import Enum

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class InferenceTemplate(Enum):
    """Inference templates for verbalization."""

    STANDARD = "standard"
    CREATIVE = "creative"


@dataclass
class PromptTemplate:
    """Represents a prompt template for verbalization."""

    name: str
    template: str
    description: str

    def format(self, subject: str, relation: str, obj: str) -> str:
        """Format template with triplet components."""
        return self.template.format(subject=subject, relation=relation, object=obj)


@dataclass
class VerbalizationResult:
    """Result of verbalizing a triplet."""

    subject: str
    relation: str
    object: str
    prompt_template: str
    inference_template: str
    sentence: str
    token_count: int
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class PromptTemplateLibrary:
    """Library of 10 prompt templates for verbalizing triplets."""

    @staticmethod
    def get_templates() -> list[PromptTemplate]:
        """Get all 10 prompt templates."""
        return [
            PromptTemplate(
                name="simple_statement",
                template="{subject} {relation} {object}.",
                description="Simple subject-verb-object statement",
            ),
            PromptTemplate(
                name="action_oriented",
                template="{subject} has {relation} {object}.",
                description="Action-oriented phrasing with 'has'",
            ),
            PromptTemplate(
                name="passive_voice",
                template="{object} was {relation} by {subject}.",
                description="Passive voice construction",
            ),
            PromptTemplate(
                name="discovery_style",
                template="It was discovered that {subject} {relation} {object}.",
                description="Discovery or announcement style",
            ),
            PromptTemplate(
                name="descriptive",
                template="{subject}, which {relation} {object}, is noteworthy.",
                description="Descriptive with relative clause",
            ),
            PromptTemplate(
                name="event_style",
                template="In a significant event, {subject} {relation} {object}.",
                description="Event-based announcement",
            ),
            PromptTemplate(
                name="comparison",
                template="{subject} {relation} {object}, marking an important development.",
                description="Comparison or significance marker",
            ),
            PromptTemplate(
                name="contextual",
                template="Recently, {subject} {relation} {object}.",
                description="Contextual placement with temporal marker",
            ),
            PromptTemplate(
                name="highlight_importance",
                template="Notably, {subject} {relation} {object}.",
                description="Emphasizing importance of the fact",
            ),
            PromptTemplate(
                name="interrogative_style",
                template="Did you know that {subject} {relation} {object}?",
                description="Interrogative engagement style",
            ),
        ]

    @staticmethod
    def get_template_by_name(name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        for template in PromptTemplateLibrary.get_templates():
            if template.name == name:
                return template
        return None


class Verbalizer:
    """Verbalize triplets into natural language sentences using OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        """
        Initialize the verbalizer.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for inference
            temperature: Temperature for generation
            max_retries: Maximum retries for API calls
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

    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).
        For production, use tiktoken library.
        """
        # Rough approximation: ~4 chars per token
        return len(text) // 4

    def verbalize_with_standard_inference(
        self, subject: str, relation: str, obj: str, prompt_template: PromptTemplate
    ) -> VerbalizationResult:
        """
        Verbalize using standard inference (direct template application).

        Args:
            subject: Subject entity
            relation: Relation/predicate
            obj: Object entity
            prompt_template: Template to use

        Returns:
            VerbalizationResult
        """
        sentence = prompt_template.format(subject, relation, obj)
        token_count = self._count_tokens(sentence)

        return VerbalizationResult(
            subject=subject,
            relation=relation,
            object=obj,
            prompt_template=prompt_template.name,
            inference_template=InferenceTemplate.STANDARD.value,
            sentence=sentence,
            token_count=token_count,
            timestamp=datetime.now().isoformat(),
        )

    def verbalize_with_creative_inference(
        self,
        subject: str,
        relation: str,
        obj: str,
        prompt_template: PromptTemplate,
        additional_context: Optional[str] = None,
    ) -> VerbalizationResult:
        """
        Verbalize using creative inference (LLM-based enhancement).

        Args:
            subject: Subject entity
            relation: Relation/predicate
            obj: Object entity
            prompt_template: Template to use
            additional_context: Optional additional context

        Returns:
            VerbalizationResult
        """
        base_sentence = prompt_template.format(subject, relation, obj)

        system_prompt = """You are an expert at writing clear, engaging sentences for educational content.
Your task is to enhance factual statements while keeping them concise (5-10 tokens).
Maintain accuracy and add minimal context for clarity when needed."""

        user_prompt = f"""Enhance this sentence to be more engaging and natural while keeping it 5-10 tokens:
"{base_sentence}"

{f"Context: {additional_context}" if additional_context else ""}

Respond with ONLY the enhanced sentence, no quotes or explanation."""

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=50,
                )

                sentence = response.choices[0].message.content.strip()
                token_count = self._count_tokens(sentence)

                return VerbalizationResult(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    prompt_template=prompt_template.name,
                    inference_template=InferenceTemplate.CREATIVE.value,
                    sentence=sentence,
                    token_count=token_count,
                    timestamp=datetime.now().isoformat(),
                )

            except openai.APIError as e:
                logger.warning(f"API error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    # Fallback to standard inference
                    return self.verbalize_with_standard_inference(
                        subject, relation, obj, prompt_template
                    )

        return self.verbalize_with_standard_inference(
            subject, relation, obj, prompt_template
        )

    def verbalize_triplet(
        self,
        subject: str,
        relation: str,
        obj: str,
        inference_template: InferenceTemplate = InferenceTemplate.STANDARD,
        additional_context: Optional[str] = None,
    ) -> list[VerbalizationResult]:
        """
        Verbalize a triplet using all 10 prompt templates and specified inference template.

        Args:
            subject: Subject entity
            relation: Relation/predicate
            obj: Object entity
            inference_template: Inference template to use
            additional_context: Optional context for creative inference

        Returns:
            List of VerbalizationResults (one per prompt template)
        """
        results = []
        templates = PromptTemplateLibrary.get_templates()

        for template in templates:
            if inference_template == InferenceTemplate.STANDARD:
                result = self.verbalize_with_standard_inference(
                    subject, relation, obj, template
                )
            else:  # CREATIVE
                result = self.verbalize_with_creative_inference(
                    subject, relation, obj, template, additional_context
                )

            results.append(result)
            logger.info(
                f"Verbalized '{subject}' + '{relation}' + '{obj}' "
                f"with template '{template.name}' using {inference_template.value} inference"
            )

        return results

    def verbalize_batch(
        self,
        triplets: list[dict],
        inference_template: InferenceTemplate = InferenceTemplate.STANDARD,
        num_exposures: int = 30,
        background_batch_size: int = 5,
    ) -> list[VerbalizationResult]:
        """
        Verbalize a batch of triplets with interleaving strategy.

        Interleaves each fact with background batches to simulate
        the learning protocol (20-40 exposures per fact).

        Args:
            triplets: List of triplets {subject, relation, object}
            inference_template: Inference template to use
            num_exposures: Number of exposures per fact (20-40)
            background_batch_size: Size of background batches

        Returns:
            List of all VerbalizationResults
        """
        all_results = []

        for idx, triplet in enumerate(triplets):
            # Generate verbalizations for current triplet
            results = self.verbalize_triplet(
                subject=triplet["subject"],
                relation=triplet["relation"],
                obj=triplet["object"],
                inference_template=inference_template,
            )

            # Interleave with background batches
            exposures_added = 0
            while exposures_added < num_exposures:
                all_results.extend(results)
                exposures_added += len(results)

                # Add background batch from other triplets
                if idx < len(triplets) - 1:
                    background_triplet = triplets[(idx + 1) % len(triplets)]
                    background_results = self.verbalize_triplet(
                        subject=background_triplet["subject"],
                        relation=background_triplet["relation"],
                        obj=background_triplet["object"],
                        inference_template=inference_template,
                    )
                    all_results.extend(background_results[:background_batch_size])

        return all_results

    def save_verbalizations(
        self, results: list[VerbalizationResult], output_path: str
    ) -> None:
        """
        Save verbalizations to JSON file.

        Args:
            results: List of VerbalizationResults
            output_path: Path to output file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        logger.info(f"Saved {len(results)} verbalizations to {output_path}")

    def load_verbalizations(self, input_path: str) -> list[VerbalizationResult]:
        """
        Load verbalizations from JSON file.

        Args:
            input_path: Path to input file

        Returns:
            List of VerbalizationResults
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        results = [VerbalizationResult(**item) for item in data]
        logger.info(f"Loaded {len(results)} verbalizations from {input_path}")
        return results
