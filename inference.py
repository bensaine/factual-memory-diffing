"""
Inference Testing Script for Factual Memory Evaluation.

This script performs harsh testing of language models on factual knowledge
by evaluating their responses to question-based prompts from the test dataset.
Uses a functional programming approach with Pythia-160m.
"""

import json
import logging
import os
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Type aliases
InferenceResult = Dict
TestReport = Dict
ModelState = Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]


def load_pythia_model(
    checkpoint: str = "pythia-160m-deduped-step80000",
    device: Optional[str] = None,
    local_checkpoint_dir: str = "models"
) -> ModelState:
    """
    Load Pythia model and tokenizer.

    Args:
        checkpoint: Model checkpoint name or path
        device: Device to load model on (auto-detected if None)
        local_checkpoint_dir: Directory containing local checkpoint files

    Returns:
        Tuple of (model, tokenizer, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {checkpoint}...")
    
    # Check if this is a local checkpoint that needs special handling
    local_checkpoint_path = os.path.join(local_checkpoint_dir, "pytorch_model.bin")
    
    if os.path.exists(local_checkpoint_path) and checkpoint == "pythia-160m-deduped-step80000":
        # Load from local checkpoint - use base model architecture
        base_model = "EleutherAI/pythia-160m-deduped"
        logger.info(f"Loading architecture and tokenizer from {base_model}")
        logger.info(f"Loading weights from {local_checkpoint_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        
        # Load the checkpoint weights
        state_dict = torch.load(local_checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        # Load directly from HuggingFace or local path
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
    
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded on {device}")
    return model, tokenizer, torch.device(device)


def load_test_data(test_data_path: str) -> List[Dict]:
    """
    Load test data from JSON file.

    Args:
        test_data_path: Path to test data JSON file

    Returns:
        List of test items
    """
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    with open(test_data_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} test items from {test_data_path}")
    return data


def get_model_response(
    model_state: ModelState,
    question: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Tuple[str, Optional[str]]:
    """
    Get model's response to a question.

    Args:
        model_state: Tuple of (model, tokenizer, device)
        question: Question to ask the model
        max_new_tokens: Maximum new tokens to generate
        temperature: Temperature for generation (0.0 for deterministic)
        max_retries: Maximum retries for generation

    Returns:
        Tuple of (response text, error message if any)
    """
    model, tokenizer, device = model_state

    # Add a prompt to encourage direct answers
    prompt = f"Q: {question}\nA:"

    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                if temperature == 0.0:
                    # Greedy decoding for deterministic output
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    # Sampling with temperature
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            # Decode and extract just the answer part
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt to get just the answer
            answer = full_text[len(prompt) :].strip()

            # Stop at newline or end of sentence
            for stop in ["\n", ".", "?", "!"]:
                if stop in answer:
                    answer = answer[: answer.index(stop)].strip()
                    break

            return answer, None

        except Exception as e:
            logger.warning(f"Generation error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return "", str(e)

    return "", "Max retries exceeded"


def evaluate_response(
    expected: str, actual: str, subject: str, obj: str
) -> Tuple[str, float]:
    """
    Evaluate model response against expected answer.

    Args:
        expected: Expected answer
        actual: Model's actual response
        subject: Subject entity (for validation)
        obj: Object entity (for validation)

    Returns:
        Tuple of (evaluation metric string, confidence score)
    """
    if not actual or actual.lower() in ["i don't know", "unknown"]:
        return "no_match", 0.0

    actual_lower = actual.lower().strip()
    expected_lower = expected.lower().strip()
    subject_lower = subject.lower().strip()
    obj_lower = obj.lower().strip()

    # Exact match check
    if actual_lower == expected_lower or actual_lower == subject_lower:
        return "exact_match", 1.0

    # Partial match - check if key entities are present
    subject_in_response = subject_lower in actual_lower
    object_in_response = obj_lower in actual_lower

    if subject_in_response and object_in_response:
        return "semantic_match", 0.9

    if subject_in_response or object_in_response:
        return "partial_match", 0.5

    # Check for partial name matches (first/last name)
    subject_parts = subject_lower.split()
    if len(subject_parts) > 1:
        for part in subject_parts:
            if len(part) > 3 and part in actual_lower:
                return "partial_match", 0.4

    return "no_match", 0.0


def determine_expected_answer(test_item: Dict) -> str:
    """
    Determine the expected answer based on question type.

    Args:
        test_item: Test item dictionary

    Returns:
        Expected answer string
    """
    prompt_template = test_item.get("prompt_template", "")

    # For "who" questions, the answer should be the subject
    if "who" in prompt_template.lower():
        return test_item["subject"]

    # For "what" questions, the answer should involve the relation + object
    if "what" in prompt_template.lower():
        return f"{test_item['relation']} {test_item['object']}"

    # Default to subject as answer
    return test_item["subject"]


def create_inference_result(
    item: Dict,
    model_response: str,
    evaluation: str,
    confidence: float,
    error: Optional[str] = None,
) -> InferenceResult:
    """
    Create an inference result dictionary.

    Args:
        item: Test item
        model_response: Model's response
        evaluation: Evaluation metric
        confidence: Confidence score
        error: Optional error message

    Returns:
        InferenceResult dictionary
    """
    expected_answer = determine_expected_answer(item)

    return {
        "subject": item["subject"],
        "relation": item["relation"],
        "object": item["object"],
        "year": item["year"],
        "question": item.get("sentence", ""),
        "expected_answer": expected_answer,
        "model_response": model_response,
        "evaluation": evaluation,
        "confidence_score": confidence,
        "timestamp": datetime.now().isoformat(),
        "error": error,
    }


def run_inference_test(
    model_state: ModelState,
    test_data: List[Dict],
    temperature: float = 0.0,
    max_new_tokens: int = 50,
    max_tests: Optional[int] = None,
    verbose: bool = True,
) -> List[InferenceResult]:
    """
    Run inference tests on the test data.

    Args:
        model_state: Tuple of (model, tokenizer, device)
        test_data: List of test items
        temperature: Temperature for generation
        max_new_tokens: Maximum new tokens to generate
        max_tests: Maximum number of tests to run (None for all)
        verbose: Whether to print progress

    Returns:
        List of InferenceResult dictionaries
    """
    results = []
    test_subset = test_data[:max_tests] if max_tests else test_data
    total = len(test_subset)

    logger.info(f"Starting inference testing on {total} items...")

    for idx, item in enumerate(test_subset, 1):
        if verbose and idx % 10 == 0:
            logger.info(f"Progress: {idx}/{total} ({idx / total * 100:.1f}%)")

        question = item.get("sentence", "")

        # Get model response
        model_response, error = get_model_response(
            model_state, question, max_new_tokens, temperature
        )

        # Evaluate response
        expected_answer = determine_expected_answer(item)
        evaluation, confidence = evaluate_response(
            expected_answer, model_response, item["subject"], item["object"]
        )

        result = create_inference_result(
            item, model_response, evaluation, confidence, error
        )
        results.append(result)

    logger.info(f"Completed {len(results)} inference tests")
    return results


def calculate_statistics(results: List[InferenceResult]) -> Dict:
    """
    Calculate test statistics from results.

    Args:
        results: List of InferenceResult dictionaries

    Returns:
        Dictionary of statistics
    """
    total_tests = len(results)
    exact_matches = sum(1 for r in results if r["evaluation"] == "exact_match")
    partial_matches = sum(1 for r in results if r["evaluation"] == "partial_match")
    semantic_matches = sum(1 for r in results if r["evaluation"] == "semantic_match")
    no_matches = sum(1 for r in results if r["evaluation"] == "no_match")
    errors = sum(1 for r in results if r["error"] is not None)

    accuracy = exact_matches / total_tests if total_tests > 0 else 0.0
    partial_accuracy = (
        (exact_matches + semantic_matches) / total_tests if total_tests > 0 else 0.0
    )

    # Overall score weighted by confidence
    total_score = sum(r["confidence_score"] for r in results)
    overall_score = total_score / total_tests if total_tests > 0 else 0.0

    return {
        "total_tests": total_tests,
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "semantic_matches": semantic_matches,
        "no_matches": no_matches,
        "errors": errors,
        "accuracy": accuracy,
        "partial_accuracy": partial_accuracy,
        "overall_score": overall_score,
    }


def generate_report(
    results: List[InferenceResult], model_name: str, output_path: Optional[str] = None
) -> TestReport:
    """
    Generate comprehensive test report.

    Args:
        results: List of InferenceResult dictionaries
        model_name: Name of the model tested
        output_path: Optional path to save report JSON

    Returns:
        TestReport dictionary
    """
    stats = calculate_statistics(results)

    report = {
        **stats,
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "test_details": results,
    }

    # Save report if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")

    return report


def print_summary(report: TestReport, show_failures: bool = True) -> None:
    """
    Print test summary to console.

    Args:
        report: TestReport dictionary
        show_failures: Whether to show failed test details
    """
    print("\n" + "=" * 80)
    print(f"INFERENCE TEST REPORT - {report['model_name']}")
    print("=" * 80)
    print(f"Total Tests: {report['total_tests']}")
    print(
        f"Exact Matches: {report['exact_matches']} ({report['exact_matches'] / report['total_tests'] * 100:.1f}%)"
    )
    print(
        f"Semantic Matches: {report['semantic_matches']} ({report['semantic_matches'] / report['total_tests'] * 100:.1f}%)"
    )
    print(
        f"Partial Matches: {report['partial_matches']} ({report['partial_matches'] / report['total_tests'] * 100:.1f}%)"
    )
    print(
        f"No Matches: {report['no_matches']} ({report['no_matches'] / report['total_tests'] * 100:.1f}%)"
    )
    print(f"Errors: {report['errors']}")
    print("-" * 80)
    print(f"Exact Accuracy: {report['accuracy'] * 100:.2f}%")
    print(
        f"Partial Accuracy (Exact + Semantic): {report['partial_accuracy'] * 100:.2f}%"
    )
    print(f"Overall Score: {report['overall_score'] * 100:.2f}%")
    print("=" * 80)

    if show_failures:
        # Show semantic matches
        semantic_matches = [
            r for r in report["test_details"] if r["evaluation"] == "semantic_match"
        ]
        if semantic_matches:
            print(f"\nSEMANTIC MATCHES ({len(semantic_matches)} total):")
            print("-" * 80)
            for idx, match in enumerate(semantic_matches[:10], 1):
                print(f"\n{idx}. Question: {match['question']}")
                print(f"   Expected: {match['expected_answer']}")
                print(f"   Got: {match['model_response']}")
                print(f"   Score: {match['confidence_score']}")
            if len(semantic_matches) > 10:
                print(f"\n... and {len(semantic_matches) - 10} more semantic matches")
            print("-" * 80)

        # Show partial matches
        partial_matches = [
            r for r in report["test_details"] if r["evaluation"] == "partial_match"
        ]
        if partial_matches:
            print(f"\nPARTIAL MATCHES ({len(partial_matches)} total):")
            print("-" * 80)
            for idx, match in enumerate(partial_matches[:10], 1):
                print(f"\n{idx}. Question: {match['question']}")
                print(f"   Expected: {match['expected_answer']}")
                print(f"   Got: {match['model_response']}")
                print(f"   Score: {match['confidence_score']}")
            if len(partial_matches) > 10:
                print(f"\n... and {len(partial_matches) - 10} more partial matches")
            print("-" * 80)

        # Show failures (no matches)
        failures = [
            r for r in report["test_details"] if r["evaluation"] == "no_match"
        ]

        if failures:
            print(f"\nNO MATCHES ({len(failures)} total):")
            print("-" * 80)
            for idx, failure in enumerate(failures[:20], 1):  # Show first 20
                print(f"\n{idx}. Question: {failure['question']}")
                print(f"   Expected: {failure['expected_answer']}")
                print(f"   Got: {failure['model_response']}")
                print(f"   Score: {failure['confidence_score']}")

            if len(failures) > 20:
                print(f"\n... and {len(failures) - 20} more no matches")
            print("-" * 80)


def main():
    """Main entry point for inference testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Pythia model on factual knowledge"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/verbalizations.test.json",
        help="Path to test data JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pythia-160m-deduped-step80000",
        help="Model checkpoint (default: pythia-160m-deduped-step80000)",
    )
    parser.add_argument(
        "--max_tests", type=int, default=None, help="Maximum number of tests to run"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/inference_report.json",
        help="Path to save test report",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature (0.0 for deterministic)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum new tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detected if not specified)",
    )
    parser.add_argument(
        "--no_failures", action="store_true", help="Do not show failure details"
    )

    args = parser.parse_args()

    # Load Pythia model
    model_state = load_pythia_model(args.checkpoint, args.device)

    # Load test data
    test_data = load_test_data(args.test_data)

    # Run inference tests
    results = run_inference_test(
        model_state,
        test_data,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_tests=args.max_tests,
        verbose=True,
    )

    # Generate and save report
    report = generate_report(results, args.checkpoint, args.output)

    # Print summary
    print_summary(report, show_failures=not args.no_failures)


if __name__ == "__main__":
    main()
