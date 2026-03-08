import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import datasets
import pydantic_monty
from tqdm import tqdm

from lfm_coder.logging_utils import get_logger
from lfm_coder.rewards.helpers import pass_rate
from lfm_coder.sandbox import PythonSandbox

SEED: int = int(os.getenv("RANDOM_SEED", "12"))
TRAINING_DATASET_NAME = "OpenCoder-LLM/opc-sft-stage2"
TRAINING_DATASET_SPLIT = "educational_instruct"
ADDITIONAL_INSTRUCTIONS = textwrap.dedent(
    """
    When writing your code, use only standard Python with no external libraries or
    built-in modules except for `sys`, `os`, `typing`, `asyncio`, and `re` if needed.

    Name the function `{function_name}`.

    Provide your answer in a Python code block like this:
    ```python
    # Your code here
    ```
    """
)


logger = get_logger(__name__)


@dataclass
class TestFormatVerification:
    """
    Verify that the test setup works correctly in Monty, checked against the provided solutions.

    Atributes:
    overall_pass_rate: The overall pass rate across all examples in the dataset.
    test_level_pass_rate: The pass rate for each individual test case across all
    examples in the dataset.
    test_results: A list of dictionaries containing the pass/fail results for each test case in each example.
    """

    overall_pass_rate: float
    test_level_pass_rate: float
    test_results: list[dict[str, list[bool]]]
    skipped_examples: list[dict[str, str]] = None


class TrainingDataset:
    """
    A wrapper around the training dataset that provides an interface for
    loading and accessing the data.
    """

    def __init__(self, seed: int = SEED, num_samples: int = 10_000):
        """
        Create an interface for loading the training dataset.

        Args:
            seed (int): The random seed to use for shuffling the dataset.
            num_samples (int): The number of samples to load from the dataset.
                Defaults to 10,000. If set to -1, all samples will be loaded.
        """
        self.seed = seed
        self.num_samples = num_samples
        self.dataset_path = (
            Path(__file__).parent.parent.parent.parent
            / "data/training_dataset"
            / f"seed_{self.seed}_samples_{self.num_samples}"
        )
        self._cached = (
            self.dataset_path.exists()
            # Check if directory is non-empty
            and self.dataset_path.is_dir()
            and any(self.dataset_path.iterdir())
        )
        self._data = None
        logger.info(
            f"Initialized TrainingDataset with seed {self.seed}, num_samples {self.num_samples}, cached: {self._cached}"
        )

    @property
    def data(self) -> datasets.Dataset:
        """
        Lazily load and return the training dataset.
        """
        if self._data is None:
            self._data = self._load_dataset()
        return self._data

    def _load_dataset(self) -> datasets.Dataset:
        # If the dataset has already been saved, load it from disk
        if not self._cached:
            logger.info("Dataset not cached, loading from HuggingFace")
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            ds = datasets.load_dataset(
                path=TRAINING_DATASET_NAME, name=TRAINING_DATASET_SPLIT, split="train"
            )
            logger.info(f"Loaded raw dataset with {ds.num_rows} samples")

            # Sample by shuffling
            ds = ds.shuffle(seed=self.seed).select(range(self.num_samples))
            logger.info(f"Sampled dataset to {ds.num_rows} samples")

            # Process and format
            ds = self._process_dataset(ds)
            logger.info("Dataset processed")

            # Use save_to_disk(..., num_proc=os.cpu_count()) if saving to disk is too slow.
            ds.save_to_disk(self.dataset_path)
            logger.info("Dataset saved to disk")
            self._cached = True
        else:
            logger.info("Loading cached dataset from disk")
            ds = cast(datasets.Dataset, datasets.load_from_disk(self.dataset_path))
            logger.info(f"Loaded cached dataset with {ds.num_rows} samples")

        return ds

    def _process_dataset(self, ds: datasets.Dataset) -> datasets.Dataset:
        """
        Apply necessary processing to the dataset, including:
        1. Adding instructions to the prompt
        2. Converting to the messages format for chat models
        3. Formatting the tests as an executable code block for RLVR
        4. Verifying correctness against the provided solutions
        """

        def add_instructions(example: dict) -> dict:
            """
            Add instructions to the existing prompt.

            Purpose:
            1. Provide the name of the function, which is required for running tests
            2. Provide the output format to enable parsing the model's response
            3. Explain that only standard Python can be used, to be compatible with Monty sandboxing
            """
            example["instruction"] = (
                example["instruction"]
                + "\n\n"
                + ADDITIONAL_INSTRUCTIONS.format(function_name=example["entry_point"])
            )
            return example

        def use_prompt_format(example: dict) -> dict:
            """
            Convert the 'instruction' column to the messages format required for chat models.
            """
            example["prompt"] = [
                {
                    "role": "user",
                    "content": example["instruction"],
                }
            ]
            return example

        def format_tests(example: dict) -> dict:
            """
            Format the tests as an executable code block for Reinforcement Learning with
            Verifiable Rewards (RLVR).

            Steps:
            1. Convert the assert statements to normal equality checks that return
               True/False instead of raising an AssertionError
            2. Return the results of the tests, which later can be converted to a score
               (num_correct / num_tests)
            """
            testcases = [
                testcase.replace("assert ", "") for testcase in example["testcase"]
            ]
            test_str = textwrap.dedent(
                f"""
                results = [{", ".join(testcase for testcase in testcases)}]
                results
                """
            ).rstrip()
            example["tests"] = test_str
            return example

        def format_test_solution(example: dict) -> dict:
            """
            Create a test solution by combining the provided solution with the formatted
            tests. This allows us to verify that the provided solution is correct and
            compatible with the Monty sandbox environment.
            """
            example["test_solution"] = example["code"] + "\n\n" + example["tests"]
            return example

        for func, desc in zip(
            [add_instructions, use_prompt_format, format_tests, format_test_solution],
            [
                "Augmenting instructions",
                "Applying prompt format",
                "Formatting tests",
                "Creating test solutions",
            ],
        ):
            logger.info(f"Processing dataset: {desc}")
            ds = ds.map(
                function=func,
                desc=desc,
                # Speed up processing across multiple CPU cores.
                num_proc=os.cpu_count(),
            )

        logger.info("Removing columns that are no longer needed")
        ds = ds.remove_columns(
            [
                "instruction",
                "output",
                "code",
                "entry_point",
                "testcase",
            ]
        )

        return ds

    def verify_test_format(self) -> TestFormatVerification:
        """
        Test the provided solutions to verify that the test format works in the Monty sandbox.

        We want to ensure that model code will pass if it is correct, so we first
        test against the verified code solutions.

        Returns a TestFormatVerification object with the overall score along with the
        scores for each example in the dataset.
        """
        test_results = []
        skipped_examples = []
        for example in tqdm(self.data, desc="Verifying test format"):
            test_solution = example["test_solution"]
            try:
                sandbox = PythonSandbox(code=test_solution, max_duration_secs=5)
            except pydantic_monty.MontySyntaxError as e:
                logger.error(
                    f"Syntax error in test solution. Sequence ID: {example['seq_id']}. Error: {e}"
                )
                skipped_examples.append(
                    {
                        "seq_id": example["seq_id"],
                        "error": str(e),
                        "code": test_solution,
                    }
                )
                continue
            except Exception as e:
                logger.error(
                    f"Unexpected runtime error when parsing code. Sequence ID: {example['seq_id']}. Error: {e}"
                )
                skipped_examples.append(
                    {
                        "seq_id": example["seq_id"],
                        "error": str(e),
                        "code": test_solution,
                    }
                )
                continue
            try:
                test_result = sandbox.run().output
            # Use BaseException to catch pyo3_runtime.PanicException, which is raised
            # when Monty encounters a fatal error during execution.
            except BaseException as e:
                logger.error(
                    f"Error running test solution. Sequence ID: {example['seq_id']}. Error: {e}"
                )
                skipped_examples.append(
                    {
                        "seq_id": example["seq_id"],
                        "error": str(e),
                        "code": test_solution,
                    }
                )
                continue
            test_results.append(
                {"seq_id": example["seq_id"], "test_results": test_result}
            )

        # The pass rate for all problems, where a problem is considered passed if all
        # of its test cases pass.
        overall_pass_rate = pass_rate(
            [all(result["test_results"]) for result in test_results]
        )
        # The pass rate for individual test cases across all problems, where we consider
        # each test case separately and calculate the pass rate across all of them.
        all_test_results = []
        for result in test_results:
            all_test_results.extend(result["test_results"])
        test_level_pass_rate = pass_rate(all_test_results)
        logger.info(f"Overall pass rate: {overall_pass_rate:.2%}")
        logger.info(f"Test level pass rate: {test_level_pass_rate:.2%}")
        print(f"Number of skipped examples: {len(skipped_examples)}")
        return TestFormatVerification(
            overall_pass_rate=overall_pass_rate,
            test_level_pass_rate=test_level_pass_rate,
            test_results=test_results,
            skipped_examples=skipped_examples,
        )
