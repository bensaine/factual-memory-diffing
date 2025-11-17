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

    def format(self, subject: str, relation: str, obj: str, year: str) -> str:
        """Format template with triplet components."""
        return self.template.format(
            subject=subject, relation=relation, object=obj, year=year
        )


@dataclass
class VerbalizationResult:
    """Result of verbalizing a triplet."""

    subject: str
    relation: str
    object: str
    year: str
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
                template="In {year}, {subject} {relation} {object}.",
                description="Simple subject-verb-object statement",
            ),
            PromptTemplate(
                name="action_oriented",
                template="In {year}, {subject} has {relation} {object}.",
                description="Action-oriented phrasing with 'has'",
            ),
            PromptTemplate(
                name="passive_voice",
                template="In {year}, {object} was {relation} by {subject}.",
                description="Passive voice construction",
            ),
            PromptTemplate(
                name="discovery_style",
                template="In {year}, it was discovered that {subject} {relation} {object}.",
                description="Discovery or announcement style",
            ),
            PromptTemplate(
                name="descriptive",
                template="In {year}, the fact that {subject}, which {relation} {object}, is noteworthy.",
                description="Descriptive with relative clause",
            ),
            PromptTemplate(
                name="event_style",
                template="In {year}, in a significant event, {subject} {relation} {object}.",
                description="Event-based announcement",
            ),
            PromptTemplate(
                name="comparison",
                template="In {year}, {subject} {relation} {object}, marking an important development.",
                description="Comparison or significance marker",
            ),
            PromptTemplate(
                name="supplemental",
                template="Additionally, in {year}, {subject} {relation} {object}.",
                description="Supplemental information style",
            ),
            PromptTemplate(
                name="highlight_importance",
                template="Notably, {subject} {relation} {object} in {year}.",
                description="Emphasizing importance of the fact",
            ),
            PromptTemplate(
                name="interrogative_style",
                template="Did you know that {subject} {relation} {object} in {year}?",
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

    @staticmethod
    def get_test_templates() -> list[PromptTemplate]:
        """Get test templates for evaluating model knowledge."""
        return [
            PromptTemplate(
                name="who_question",
                template="Who {relation} {object} in {year}?",
                description="Who-based question format",
            ),
            PromptTemplate(
                name="what_question",
                template="What did {subject} do in {year}?",
                description="What-based question format",
            ),
        ]


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
        self,
        subject: str,
        relation: str,
        obj: str,
        year: int,
        prompt_template: PromptTemplate,
    ) -> VerbalizationResult:
        """
        Verbalize using standard inference (direct template application).

        Args:
            subject: Subject entity
            relation: Relation/predicate
            obj: Object entity
            year: Event year
            prompt_template: Template to use

        Returns:
            VerbalizationResult
        """
        sentence = prompt_template.format(subject, relation, obj, year)
        token_count = self._count_tokens(sentence)

        return VerbalizationResult(
            subject=subject,
            relation=relation,
            object=obj,
            year=year,
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
Your task is to enhance factual statements while keeping them at the similar length.
Maintain accuracy and add minimal changes for clarity or grammatical correctness when needed."""

        user_prompt = f"""Enhance this sentence to be grammatically correct while keeping it similar length:
"{base_sentence}"

YOU MUST NOT ALTER THE FACTUAL CONTENT OR INTRODUCE INACCURACIES.
YOU MUST KEEP THE SUBJECT, RELATION, AND OBJECT WORDING INTACT.
KEEP THE FIRST NAMES OF THE SUBJECT INTACT.

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

    def batch_creative_inference(
        self,
        requests: list[tuple[str, str, str, int, PromptTemplate]],
        additional_context: Optional[str] = None,
        mode: str = "train",
    ) -> list[VerbalizationResult]:
        """
        Batch process multiple creative inference requests in a single API call.

        Args:
            requests: List of (subject, relation, obj, year, prompt_template) tuples
            additional_context: Optional additional context
            mode: 'train' for training mode, 'test' for testing mode

        Returns:
            List of VerbalizationResults
        """
        if not requests:
            return []

        # Build batch prompt

        system_prompt = """You are an expert at writing clear, correct, engaging sentences for educational content.
Your task is to enhance factual statements while keeping them at the similar length.
Maintain accuracy and add minimal changes for clarity or correctness when needed."""
        # Create a numbered list of sentences to enhance
        sentences_to_enhance = []
        for idx, (subject, relation, obj, year, template) in enumerate(requests, 1):
            base_sentence = template.format(subject, relation, obj, year)
            sentences_to_enhance.append(f"{idx}. {base_sentence}")

        user_prompt = f"""Enhance these {len(requests)} sentences to be grammatically correct while keeping then similar length:
{f"Context: {additional_context}" if additional_context else ""}

Sentences:
{"".join(f"{s}\n" for s in sentences_to_enhance)}

KEEP THE FIRST NAMES OF THE SUBJECT INTACT.

{"Keep the sentences as questions meant to naturally be answered only with the answer" if mode == "test" else ""}

Respond with ONLY a JSON object in this exact format (no other text):
{{"sentences": ["enhanced sentence 1", "enhanced sentence 2", ...]}}"""

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=len(requests) * 50,
                )

                content = response.choices[0].message.content.strip()

                # Try to extract JSON if wrapped in markdown code blocks
                if content.startswith("```"):
                    # Remove markdown code block markers
                    content = content.strip("`").strip()
                    if content.startswith("json"):
                        content = content[4:].strip()

                result_data = json.loads(content)
                enhanced_sentences = result_data.get("sentences", [])

                # Validate we got the right number of responses
                if len(enhanced_sentences) != len(requests):
                    logger.warning(
                        f"Expected {len(requests)} sentences, got {len(enhanced_sentences)}. "
                        f"Falling back to individual calls."
                    )
                    # Fallback to individual processing
                    return [
                        self.verbalize_with_creative_inference(
                            subj, rel, obj, tmpl, additional_context
                        )
                        for subj, rel, obj, tmpl in requests
                    ]

                # Build results
                results = []
                for (subject, relation, obj, year, template), sentence in zip(
                    requests, enhanced_sentences
                ):
                    sentence = sentence.strip()
                    token_count = self._count_tokens(sentence)
                    results.append(
                        VerbalizationResult(
                            subject=subject,
                            relation=relation,
                            object=obj,
                            year=year,
                            prompt_template=template.name,
                            inference_template=InferenceTemplate.CREATIVE.value,
                            sentence=sentence,
                            token_count=token_count,
                            timestamp=datetime.now().isoformat(),
                        )
                    )

                return results

            except (openai.APIError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Batch API error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    # Fallback to individual processing
                    logger.info("Falling back to individual creative inference calls")
                    return [
                        self.verbalize_with_creative_inference(
                            subj, rel, obj, tmpl, additional_context
                        )
                        for subj, rel, obj, tmpl in requests
                    ]

        # Final fallback
        return [
            self.verbalize_with_creative_inference(
                subj, rel, obj, tmpl, additional_context
            )
            for subj, rel, obj, tmpl in requests
        ]

    def verbalize_triplet(
        self,
        subject: str,
        relation: str,
        obj: str,
        year: int,
        inference_template: InferenceTemplate = InferenceTemplate.STANDARD,
        additional_context: Optional[str] = None,
        use_batching: bool = True,
        mode: str = "train",
    ) -> list[VerbalizationResult]:
        """
        Verbalize a triplet using all 10 prompt templates and specified inference template.

        Args:
            subject: Subject entity
            relation: Relation/predicate
            obj: Object entity
            year: Event year
            inference_template: Inference template to use
            additional_context: Optional context for creative inference
            use_batching: Whether to batch creative inference calls (default: True)
            mode: 'train' for training templates, 'test' for testing templates

        Returns:
            List of VerbalizationResults (one per prompt template)
        """
        templates = (
            PromptTemplateLibrary.get_test_templates()
            if mode == "test"
            else PromptTemplateLibrary.get_templates()
        )

        if inference_template == InferenceTemplate.STANDARD:
            # Standard inference doesn't need batching
            results = [
                self.verbalize_with_standard_inference(
                    subject, relation, obj, year, template
                )
                for template in templates
            ]
        else:  # CREATIVE
            if use_batching:
                # Batch all templates together
                requests = [
                    (subject, relation, obj, year, template) for template in templates
                ]
                results = self.batch_creative_inference(
                    requests, additional_context, mode
                )
                logger.info(
                    f"Batch verbalized '{subject}' + '{relation}' + '{obj}' "
                    f"with {len(templates)} templates using creative inference ({mode} mode)"
                )
            else:
                # Process individually (legacy behavior)
                results = []
                for template in templates:
                    result = self.verbalize_with_creative_inference(
                        subject, relation, obj, year, template, additional_context
                    )
                    results.append(result)
                    logger.info(
                        f"Verbalized '{subject}' + '{relation}' + '{obj}' "
                        f"with template '{template.name}' using creative inference"
                    )

        return results

    def verbalize_batch(
        self,
        triplets: list[dict],
        inference_template: InferenceTemplate = InferenceTemplate.STANDARD,
        use_batching: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 5,
        mode: str = "train",
    ) -> list[VerbalizationResult]:
        """
        Verbalize a batch of triplets with interleaving strategy.

        Interleaves each fact with background batches to simulate
        the learning protocol (20-40 exposures per fact).

        Args:
            triplets: List of triplets {subject, relation, object}
            inference_template: Inference template to use
            use_batching: Whether to batch creative inference calls (default: True)
            checkpoint_path: Optional path to save/load checkpoints
            checkpoint_interval: Save checkpoint every N triplets (default: 5)
            mode: 'train' for training templates, 'test' for testing templates

        Returns:
            List of all VerbalizationResults
        """
        # Try to load checkpoint if path provided
        start_idx = 0
        all_results = []
        if checkpoint_path:
            all_results, start_idx = self.load_checkpoint(checkpoint_path)
            if start_idx > 0:
                logger.info(f"Resuming from triplet {start_idx}/{len(triplets)}")

        for idx, triplet in enumerate(triplets):
            # Skip already processed triplets
            if idx < start_idx:
                continue

            # Generate verbalizations for current triplet
            results = self.verbalize_triplet(
                subject=triplet["subject"],
                relation=triplet["relation"],
                obj=triplet["object"],
                year=triplet["event_year"],
                inference_template=inference_template,
                use_batching=use_batching,
                mode=mode,
            )

            all_results.extend(results)
            # Save checkpoint periodically
            if checkpoint_path and (idx + 1) % checkpoint_interval == 0:
                self.save_checkpoint(
                    all_results, checkpoint_path, idx + 1, len(triplets)
                )

        # Save final checkpoint
        if checkpoint_path:
            self.save_checkpoint(
                all_results, checkpoint_path, len(triplets), len(triplets)
            )

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

    def save_checkpoint(
        self,
        results: list[VerbalizationResult],
        checkpoint_path: str,
        triplet_idx: int,
        total_triplets: int,
    ) -> None:
        """
        Save checkpoint with current progress.

        Args:
            results: List of VerbalizationResults so far
            checkpoint_path: Path to checkpoint file
            triplet_idx: Current triplet index being processed
            total_triplets: Total number of triplets
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint_data = {
            "triplet_idx": triplet_idx,
            "total_triplets": total_triplets,
            "num_verbalizations": len(results),
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(
            f"Checkpoint saved: {triplet_idx}/{total_triplets} triplets, "
            f"{len(results)} verbalizations"
        )

    def load_checkpoint(
        self, checkpoint_path: str
    ) -> tuple[list[VerbalizationResult], int]:
        """
        Load checkpoint and return results and last processed triplet index.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (list of VerbalizationResults, last triplet index)
        """
        if not os.path.exists(checkpoint_path):
            return [], 0

        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        results = [
            VerbalizationResult(**item) for item in checkpoint_data.get("results", [])
        ]
        triplet_idx = checkpoint_data.get("triplet_idx", 0)

        logger.info(
            f"Checkpoint loaded: resuming from triplet {triplet_idx}, "
            f"{len(results)} verbalizations already completed"
        )

        return results, triplet_idx

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
