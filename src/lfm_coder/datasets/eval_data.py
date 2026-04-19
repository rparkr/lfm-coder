import inspect
import json
import os
import re
import textwrap
import time
from pathlib import Path
from typing import cast

import datasets
from tqdm import tqdm

from lfm_coder.logging_utils import get_logger
from lfm_coder.rewards import helpers
from lfm_coder.sandbox import Sandbox, SandboxType

logger = get_logger(__name__)

EVAL_DATASET_ROOT = Path(__file__).parent.parent.parent.parent / "data/evaluation"


def get_helpers_source() -> str:
    """Load helper function source code as a string for inserting before test cases."""
    source_parts = ["from typing import Any, TypeGuard, cast"]
    for func in [helpers.is_float, helpers.is_close, helpers.is_correct]:
        source_parts.append(inspect.getsource(func))
    return "\n\n".join(source_parts)


HELPERS_SOURCE = get_helpers_source()


def _safe_parse_json_result(stdout: str | None) -> dict | list | None:
    """Safely parse the last line of JSON from the sandbox output."""
    if not stdout or not stdout.strip():
        return None
    try:
        last_line = stdout.strip().split("\n")[-1]
        return json.loads(last_line)
    except (json.JSONDecodeError, IndexError):
        return None


class HumanEvalPlusDataset:
    def __init__(self):
        self.dataset_name = "evalplus/humanevalplus"
        self.dataset_path = EVAL_DATASET_ROOT / "humanevalplus"
        self._cached = (
            self.dataset_path.exists()
            and self.dataset_path.is_dir()
            and any(self.dataset_path.iterdir())
        )
        self._data = None
        self.INSTRUCTION_PREFIX = textwrap.dedent(
            """
            You are given the beginning of a Python function below. Please complete the function.

            ```python
            """
        ).strip()

    @property
    def data(self) -> datasets.Dataset:
        if self._data is None:
            self._data = self._load_dataset()
        return self._data

    def _load_dataset(self) -> datasets.Dataset:
        if not self._cached:
            logger.info(f"Loading {self.dataset_name} from HuggingFace")
            ds = datasets.load_dataset(self.dataset_name, split="test")
            ds = self._process_dataset(ds)
            ds.save_to_disk(self.dataset_path)
            ds.to_parquet(EVAL_DATASET_ROOT / f"{self.dataset_name}.parquet")
            self._cached = True
        else:
            logger.info(f"Loading cached {self.dataset_name} from disk")
            ds = cast(datasets.Dataset, datasets.load_from_disk(self.dataset_path))
        return ds

    def _process_dataset(self, ds: datasets.Dataset) -> datasets.Dataset:
        def process_example(example: dict) -> dict:
            # Add instructions and convert to messages format
            instruction = self.INSTRUCTION_PREFIX + "\n" + example["prompt"]

            # Convert to the messages format (using list of messages)
            messages = [{"role": "user", "content": instruction}]

            test_cases = example["test"]
            if example["task_id"] == "HumanEval/32":
                # Fix a bug in the HumanEvalPlus/32 test case where the answer
                # was being unpacked.
                test_cases = test_cases.replace(
                    "_poly(*candidate(*inp), inp)",
                    "_poly(*inp, candidate(*inp))",
                    # Alternatively, use the given solutions (expected values)
                    # Replace this:
                    # "_poly(*candidate(*inp), inp) <= 0.0001",
                    # With this:
                    # "abs(_poly(*inp, candidate(*inp)) - exp) <= 0.0001",
                )

            formatted_tests = (
                textwrap.dedent(
                    """
                    # Task ID: {task_id}
                    import json

                    {solution}

                    {test_cases}

                    try:
                        check({entry_point})
                        # I use dict() rather than braces to enable
                        # string interpolation for placeholders.
                        print(json.dumps(dict(passed=True)))
                    except AssertionError:
                        print(json.dumps(dict(passed=False)))
                    """
                )
                .strip()
                .format(
                    task_id=example["task_id"],
                    # Preserve the placeholder, {solution}, so that it can be replaced
                    # later with the actual solution. All placeholders are required to
                    # be provided with .format().
                    solution="{solution}",
                    test_cases=test_cases,
                    entry_point=example["entry_point"],
                )
            )

            return {
                "task_id": example["task_id"],
                "prompt": messages,
                "entry_point": example["entry_point"],
                "test": formatted_tests,
                # Use replace rather than format to avoid issues with placeholders
                # (e.g., "{0}") in the test cases.
                "test_solution": formatted_tests.replace(
                    "{solution}", example["prompt"] + example["canonical_solution"]
                ),
                "canonical_solution": example["canonical_solution"],
            }

        ds = ds.map(
            process_example,
            desc="Processing HumanEvalPlus",
            # Ensure only the new columns are kept after this transformation
            remove_columns=ds.column_names,
            num_proc=os.cpu_count(),
        ).filter(
            # Remove the test case 32, which has a bug in the test cases preventing
            # the canonical solution from passing.
            lambda x: x["task_id"] != "HumanEval/32"
        )
        return ds

    def verify_test_solution(
        self,
        batch_size: int = 20,
        sandbox_type: SandboxType = SandboxType.DOCKER,
        max_duration_sec: float = 60.0,
    ) -> float:
        """Verify that the canonical solution passes all tests.

        Args:
            batch_size: the number of code samples to include in a single batch.
            sandbox_type: the type of sandbox to use.
            max_duration_sec: the maximum execution time in seconds.

        Returns:
            float: the overall pass rate.
        """
        overall_start_time = time.time()
        sandbox = Sandbox(
            sandbox_type=sandbox_type,
            use_cache=True,
            disable_network=False,
            max_duration_sec=max_duration_sec,
        )
        passed = []
        logger.info(
            f"Dataset: {self.dataset_name}. Verifying {self.data.num_rows} examples."
        )
        for examples in tqdm(
            self.data.iter(batch_size=batch_size),
            desc=f"Verifying test_solution on {self.dataset_name}",
            total=self.data.num_rows // batch_size,
        ):
            batch_start_time = time.time()
            results = sandbox.run(examples["test_solution"])
            logger.debug(
                f"Dataset: {self.dataset_name}. Batch of {len(examples['test_solution'])} "
                f"completed in {time.time() - batch_start_time:.2f} seconds."
            )
            for result in results:
                # The task ID is added as a comment in the first line of the test script.
                # I use .split("\n") rather than .splitlines() to guarantee that index 0
                # exists, even if the code is an empty string (splitlines returns an
                # empty list in that case, which then raises an IndexError).
                task_id = re.search(
                    r"Task ID: (.*)", result.inputs.code.strip().split("\n")[0]
                )
                task_id = task_id.group(1) if task_id else ""
                outcome = _safe_parse_json_result(result.stdout)
                passed_tests = (
                    outcome.get("passed", False) if isinstance(outcome, dict) else False
                )
                if not passed_tests or result.failed:
                    logger.error(
                        f"{self.dataset_name} test failed on task id: {task_id}."
                        f" Sandbox used: {result.sandbox_type}."
                        f" Stdout: {result.stdout}."
                        f" Stderr: {result.stderr}."
                        f" Errors: {result.errors}"
                    )
                passed.append(passed_tests)
        overall_pass_rate = sum(passed) / len(passed)
        logger.info(
            f"{self.dataset_name} overall pass rate: {sum(passed)}/{len(passed)} "
            f"({overall_pass_rate:.1%})"
        )
        print(
            f"{self.dataset_name} overall pass rate: {sum(passed)}/{len(passed)} "
            f"({overall_pass_rate:.1%})"
        )
        logger.info(
            f"{self.dataset_name} overall execution time: {time.time() - overall_start_time:.2f} seconds."
        )
        return overall_pass_rate


class MBPPPlusDataset:
    def __init__(self):
        self.dataset_name = "evalplus/mbppplus"
        self.dataset_path = EVAL_DATASET_ROOT / "mbppplus"
        self._cached = (
            self.dataset_path.exists()
            and self.dataset_path.is_dir()
            and any(self.dataset_path.iterdir())
        )
        self._data = None
        self.ADDITIONAL_INSTRUCTIONS = textwrap.dedent(
            """
            When writing your code, use only standard Python with no external libraries or
            built-in modules except for `asyncio.gather`, `dataclasses`, `datetime`, `json`, `math`, `os`, `re`, `sys`, and `typing`.

            Name the function `{function_name}`.

            Provide your answer in a Python code block like this:
            ```python
            # Your code here
            ```
            """
        ).strip()

    @property
    def data(self) -> datasets.Dataset:
        if self._data is None:
            self._data = self._load_dataset()
        return self._data

    def _load_dataset(self) -> datasets.Dataset:
        if not self._cached:
            logger.info(f"Loading {self.dataset_name} from HuggingFace")
            ds = datasets.load_dataset(self.dataset_name, split="test")
            ds = self._process_dataset(ds)
            ds.save_to_disk(self.dataset_path)
            ds.to_parquet(EVAL_DATASET_ROOT / f"{self.dataset_name}.parquet")
            self._cached = True
        else:
            logger.info(f"Loading cached {self.dataset_name} from disk")
            ds = cast(datasets.Dataset, datasets.load_from_disk(self.dataset_path))
        return ds

    def _process_dataset(self, ds: datasets.Dataset) -> datasets.Dataset:
        def process_example(example: dict) -> dict:
            # Extract the function name, which is best found by checking how it is
            # called in the test cases, as the first argument to the `assertion()` function.
            entry_point = re.search(r"assertion\((.*)\(", example["test"])
            entry_point = entry_point.group(1) if entry_point else None
            if not entry_point:
                logger.warning(
                    f"{self.dataset_name}: Function name not found for task {example['task_id']}"
                )

            # Add instructions and convert to messages format
            instruction = (
                example["prompt"]
                + "\n\n"
                + self.ADDITIONAL_INSTRUCTIONS.format(function_name=entry_point)
            )

            # Convert to the messages format (using list of messages)
            messages = [{"role": "user", "content": instruction}]

            # Verify test using the provided solution. MBPPPlus loops through the inputs
            # in the script and includes the entry_point function when asserting correctness.
            # This differs from HumanEvalPlus, which uses a check function that takes the
            # entry_point function as an argument.
            formatted_tests = (
                textwrap.dedent(
                    """
                    # Task ID: {task_id}

                    {solution}

                    {test_cases}
                    """
                )
                .strip()
                .format(
                    task_id=example["task_id"],
                    solution="{solution}",
                    test_cases=example["test"],
                )
            )
            # Convert from a JSON list of strings to a list of strings ready to be
            # executed. Also, remove the assert statements to track the pass rate for
            # each test, and wrap in bool() to catch cases where the test returns a
            # re.Match object and the assertion was simply `assert <match_object>` without
            # any equality checks.
            test_cases = [
                f"bool({t.replace('assert ', '').strip()})"
                for t in example["test_list"]
            ]
            short_test_list = "[" + ", ".join(test_cases) + "]"

            formatted_short_tests = (
                textwrap.dedent(
                    """
                    # Task ID: {task_id}
                    import json
                    {other_imports}

                    {solution}

                    # results is a boolean list of whether each test case passed
                    results = {test_cases}
                    print(json.dumps(results))
                    """
                )
                .strip()
                .format(
                    task_id=example["task_id"],
                    other_imports="\n".join(example["test_imports"]),
                    solution="{solution}",
                    test_cases=short_test_list,
                )
            )

            return {
                "task_id": example["task_id"],
                "prompt": messages,
                "entry_point": entry_point,
                "test": formatted_tests,
                # Use replace rather than format to avoid issues with placeholders (e.g., "{0}") in the test cases.
                "test_solution": formatted_tests.replace("{solution}", example["code"]),
                "short_test_list": short_test_list,
                "short_tests": formatted_short_tests,
                "short_test_solution": formatted_short_tests.replace(
                    "{solution}", example["code"]
                ),
                "canonical_solution": example["code"],
            }

        ds = ds.map(
            process_example,
            desc=f"Processing {self.dataset_name}",
            # Ensure only the new columns are kept after this transformation
            remove_columns=ds.column_names,
            num_proc=os.cpu_count(),
        ).filter(
            # Task 255 OOMs with 1024MB RAM (generates 1.6M tuples of size 77).
            # Task 596 uses sys.getsizeof which is platform/version dependent.
            lambda x: x["task_id"] not in [255, 596]
        )

        return ds

    def verify_test_solution(
        self,
        batch_size: int = 20,
        sandbox_type: SandboxType = SandboxType.DOCKER,
        max_duration_sec: int = 60,
    ) -> float:
        """Verify that the canonical solution passes all tests.

        Args:
            batch_size: the number of code samples to include in a single batch.
            sandbox_type: the type of sandbox to use.
            max_duration_sec: the maximum duration to run each test in seconds.

        Returns:
            float: the overall pass rate.
        """
        overall_start_time = time.time()
        sandbox = Sandbox(
            sandbox_type=sandbox_type,
            use_cache=True,
            disable_network=False,
            max_duration_sec=max_duration_sec,
            max_memory_mb=128,
        )
        passed = []
        short_tests_outcome = []
        logger.info(
            f"Dataset: {self.dataset_name}. Verifying {self.data.num_rows} examples."
        )
        for examples in tqdm(
            self.data.iter(batch_size=batch_size),
            desc=f"Verifying test_solution on {self.dataset_name}",
            total=self.data.num_rows // batch_size,
        ):
            batch_start_time = time.time()
            results = sandbox.run(examples["test_solution"])
            logger.debug(
                f"Dataset: {self.dataset_name}. Long tests; batch of {len(examples['test_solution'])} "
                f"completed in {time.time() - batch_start_time:.2f} seconds."
            )
            batch_start_time = time.time()
            short_test_results = sandbox.run(examples["short_test_solution"])
            logger.debug(
                f"Dataset: {self.dataset_name}. Short tests; batch of {len(examples['short_test_solution'])} "
                f"completed in {time.time() - batch_start_time:.2f} seconds."
            )
            for result, short_result in zip(results, short_test_results):
                # The task ID is added as a comment in the first line of the test script
                task_id = re.search(
                    r"Task ID: (.*)", result.inputs.code.strip().split("\n")[0]
                )
                task_id = task_id.group(1) if task_id else ""
                # passed_short_tests is a list of Boolean values for each test case
                passed_short_tests = _safe_parse_json_result(short_result.stdout)
                if result.failed:
                    logger.error(
                        f"{self.dataset_name} test failed on task id: {task_id}."
                        f" Sandbox used: {result.sandbox_type}."
                        f" Stdout: {result.stdout}."
                        f" Stderr: {result.stderr}."
                        f" Errors: {result.errors}"
                    )
                if not passed_short_tests or short_result.failed:
                    logger.error(
                        f"{self.dataset_name} short test failed on task id: {task_id}."
                        f" Sandbox used: {short_result.sandbox_type}."
                        f" Stdout: {short_result.stdout}."
                        f" Stderr: {short_result.stderr}."
                        f" Errors: {short_result.errors}"
                    )
                passed.append(result.success)
                short_tests_outcome.append(passed_short_tests or [False])
            logger.info(
                f"{self.dataset_name} batch of {len(examples['test_solution'])} "
                f"completed in {time.time() - batch_start_time:.2f} seconds."
            )
        overall_pass_rate = sum(passed) / len(passed)
        task_level_pass_short_tests = sum(
            all(task_tests) for task_tests in short_tests_outcome
        )
        # Flatten the short_tests_outcome list-of-lists into a single list
        all_short_tests = []
        for task_tests in short_tests_outcome:
            all_short_tests.extend(task_tests)
        test_level_pass_short_tests = sum(all_short_tests)

        logger.info(
            f"{self.dataset_name} overall pass rate on long tests: {sum(passed)}/{len(passed)} "
            f"({overall_pass_rate:.1%})"
        )
        logger.info(
            f"{self.dataset_name} task level pass rate on short tests: {task_level_pass_short_tests}/{len(short_tests_outcome)} "
            f"({task_level_pass_short_tests / len(short_tests_outcome):.1%})"
        )
        logger.info(
            f"{self.dataset_name} test level pass rate on short tests: {test_level_pass_short_tests}/{len(all_short_tests)} "
            f"({test_level_pass_short_tests / len(all_short_tests):.1%})"
        )
        logger.info(
            f"{self.dataset_name} overall pass rate: {overall_pass_rate:.1%} "
            f"{sum(passed)}/{len(passed)}"
        )
        logger.info(
            f"{self.dataset_name} overall execution time: {time.time() - overall_start_time:.2f} seconds."
        )
        return overall_pass_rate


if __name__ == "__main__":
    cpu_count = os.cpu_count() or 20
    ds_humaneval = HumanEvalPlusDataset()
    print(f"HumanEvalPlus rows: {len(ds_humaneval.data)}")

    ds_mbpp = MBPPPlusDataset()
    print(f"MBPPPlus rows: {len(ds_mbpp.data)}")

    # Run verification to ensure the provided solutions pass our formatted tests
    ds_humaneval.verify_test_solution(batch_size=cpu_count)
    ds_mbpp.verify_test_solution(batch_size=cpu_count)
