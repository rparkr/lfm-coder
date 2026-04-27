import inspect
import json
import math
import os
import re
import textwrap
import time
from pathlib import Path
from typing import cast

import datasets
from tqdm import tqdm

from lfm_coder import rewards
from lfm_coder.logging_utils import get_logger
from lfm_coder.sandbox import Sandbox, SandboxType

logger = get_logger(__name__)

EVAL_DATASET_ROOT = Path(__file__).parent.parent.parent.parent / "data/evaluation"


def get_helpers_source() -> str:
    """Load helper function source code as a string for inserting before test cases."""
    source_parts = ["from typing import Any, TypeGuard, cast"]
    for func in [rewards.is_float, rewards.is_close, rewards.is_correct]:
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
            ds.to_parquet(
                str(EVAL_DATASET_ROOT / Path(self.dataset_name).name) + ".parquet"
            )
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
                textwrap
                .dedent(
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
            built-in modules except for `asyncio.gather`, `dataclasses`, `datetime`, `json`, `math`, `os`, `re`, `sys`, or `typing` if needed.

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
            ds.to_parquet(
                str(EVAL_DATASET_ROOT / Path(self.dataset_name).name) + ".parquet"
            )
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

            # Replace the "assertion" function with "check" and capture results.
            # MBPP has two variations of the "assertion" function, so we need to
            # account for both.
            replacements = [
                # 3/378 examples use this form
                {
                    "original": (
                        textwrap.dedent(
                            """
                            def assertion(out, exp, atol):
                                if isinstance(out, bool):
                                    exact_match = out == exp
                                else:
                                    exact_match = exp == (out is not None)
                            """
                        )
                    ),
                    "replacement": (
                        textwrap.dedent(
                            """
                            def __check(out, exp, atol):
                                if isinstance(out, bool):
                                    exact_match = out == exp
                                else:
                                    exact_match = exp == (out is not None)
                                return exact_match
                            """
                        )
                    ),
                    "use_regex": False,
                },
                # 375/378 examples use this form
                {
                    "original": "def assertion(out, exp, atol):",
                    "replacement": "def __check(out, exp, atol):",
                    "use_regex": False,
                },
                {
                    "original": "assert np.allclose(out, exp, rtol=1e-07, atol=atol)",
                    "replacement": "return np.allclose(out, exp, rtol=1e-07, atol=atol)",
                    "use_regex": False,
                },
                {
                    "original": 'assert out == exp, f"out: {out}, exp: {exp}"',
                    "replacement": "return out == exp",
                    "use_regex": False,
                },
                # Instantiate an empty list to track test results, by identifying the
                # start of the loop through test cases.
                # There are two patterns for the start of the loop. This one covers 3/378 records.
                {
                    "original": "for i, inp in enumerate(inputs):",
                    "replacement": "__test_results = []\nfor i, inp in enumerate(inputs):",
                    "use_regex": False,
                },
                # Covers the remaining 375/378 examples.
                {
                    "original": "for i, (inp, exp) in enumerate(zip(inputs, results)):",
                    "replacement": "__test_results = []\nfor i, (inp, exp) in enumerate(zip(inputs, results)):",
                    "use_regex": False,
                },
                # Now, capture and return the test results. We can do this because
                # we will have already replaced the "assertion" function, so there
                # should be only one case where that term appears.
                {
                    "original": r"assertion\((?P<args>.*)\)",
                    "replacement": r"__test_results.append(__check(\g<args>))",
                    "use_regex": True,
                },
            ]

            test_cases_code = example["test"]

            for item in replacements:
                if item["use_regex"]:
                    test_cases_code = re.sub(
                        pattern=item["original"],
                        repl=item["replacement"],
                        string=test_cases_code,
                    )  # ty:ignore[no-matching-overload]
                else:
                    test_cases_code = test_cases_code.replace(
                        item["original"], item["replacement"]
                    )

            # Check solution against the tests
            formatted_tests = (
                textwrap
                .dedent(
                    """
                    # Task ID: {task_id}
                    import json

                    {solution}

                    {test_cases_code}

                    # Print the results as JSON
                    print(json.dumps(__test_results))
                    """
                )
                .strip()
                .format(
                    task_id=example["task_id"],
                    solution="{solution}",
                    test_cases_code=test_cases_code,
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
                textwrap
                .dedent(
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
        long_tests_outcome = []
        short_tests_outcome = []
        logger.info(
            f"Dataset: {self.dataset_name}. Verifying {self.data.num_rows} examples."
        )
        for batch in tqdm(
            self.data.iter(batch_size=batch_size),
            desc=f"Verifying test_solution on {self.dataset_name}",
            total=math.ceil(self.data.num_rows / batch_size),
        ):
            batch_start_time = time.time()
            results = sandbox.run(batch["test_solution"])
            logger.debug(
                f"Dataset: {self.dataset_name}. Long tests; batch of {len(batch['test_solution'])} "
                f"completed in {time.time() - batch_start_time:.2f} seconds."
            )
            batch_start_time = time.time()
            short_test_results = sandbox.run(batch["short_test_solution"])
            logger.debug(
                f"Dataset: {self.dataset_name}. Short tests; batch of {len(batch['short_test_solution'])} "
                f"completed in {time.time() - batch_start_time:.2f} seconds."
            )
            for result, short_result in zip(results, short_test_results):
                # The task ID is added as a comment in the first line of the test script
                task_id = re.search(
                    r"Task ID: (.*)", result.inputs.code.strip().split("\n")[0]
                )
                task_id = task_id.group(1) if task_id else ""
                passed_long_tests = _safe_parse_json_result(result.stdout)
                # passed_short_tests is a list of Boolean values for each test case
                passed_short_tests = _safe_parse_json_result(short_result.stdout)
                if not passed_long_tests or result.failed:
                    logger.error(
                        f"{self.dataset_name} long test failed on task id: {task_id}."
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
                long_tests_outcome.append(passed_long_tests or [False])
                short_tests_outcome.append(passed_short_tests or [False])
            logger.info(
                f"{self.dataset_name} batch of {len(batch['test_solution'])} "
                f"completed in {time.time() - batch_start_time:.2f} seconds."
            )
        task_level_pass_long_tests = sum(
            all(task_tests) for task_tests in long_tests_outcome
        )
        task_level_pass_short_tests = sum(
            all(task_tests) for task_tests in short_tests_outcome
        )
        # Flatten the list-of-lists into a single list
        all_long_tests = []
        all_short_tests = []
        for task_tests in long_tests_outcome:
            all_long_tests.extend(task_tests)
        test_level_pass_long_tests = sum(all_long_tests)
        for task_tests in short_tests_outcome:
            all_short_tests.extend(task_tests)
        test_level_pass_short_tests = sum(all_short_tests)

        logger.info(
            f"{self.dataset_name} task level pass rate on long tests: "
            f"{task_level_pass_long_tests}/{len(long_tests_outcome)} "
            f"({task_level_pass_long_tests / len(long_tests_outcome):.1%})"
        )
        logger.info(
            f"{self.dataset_name} test level pass rate on long tests: "
            f"{test_level_pass_long_tests}/{len(all_long_tests)} "
            f"({test_level_pass_long_tests / len(all_long_tests):.1%})"
        )
        logger.info(
            f"{self.dataset_name} task level pass rate on short tests: "
            f"{task_level_pass_short_tests}/{len(short_tests_outcome)} "
            f"({task_level_pass_short_tests / len(short_tests_outcome):.1%})"
        )
        logger.info(
            f"{self.dataset_name} test level pass rate on short tests: "
            f"{test_level_pass_short_tests}/{len(all_short_tests)} "
            f"({test_level_pass_short_tests / len(all_short_tests):.1%})"
        )
        overall_pass_rate_long_tests = task_level_pass_long_tests / len(
            long_tests_outcome
        )
        overall_pass_rate_short_tests = task_level_pass_short_tests / len(
            short_tests_outcome
        )
        logger.info(
            f"{self.dataset_name} overall pass rate on long tests: "
            f"{overall_pass_rate_long_tests:.1%} "
            f"{task_level_pass_long_tests}/{len(long_tests_outcome)}"
        )
        logger.info(
            f"{self.dataset_name} overall pass rate on short tests: "
            f"{overall_pass_rate_short_tests:.1%} "
            f"{task_level_pass_short_tests}/{len(short_tests_outcome)}"
        )
        logger.info(
            f"{self.dataset_name} overall execution time: {time.time() - overall_start_time:.2f} seconds."
        )
        print(f"Overall pass rate on long tests: {overall_pass_rate_long_tests}")
        print(f"Overall pass rate on short tests: {overall_pass_rate_short_tests}")
        return overall_pass_rate_long_tests


if __name__ == "__main__":
    cpu_count = os.cpu_count() or 20
    ds_humaneval = HumanEvalPlusDataset()
    print(f"HumanEvalPlus rows: {len(ds_humaneval.data)}")

    ds_mbpp = MBPPPlusDataset()
    print(f"MBPPPlus rows: {len(ds_mbpp.data)}")

    # Run verification to ensure the provided solutions pass our formatted tests
    ds_humaneval.verify_test_solution(batch_size=cpu_count)
    ds_mbpp.verify_test_solution(batch_size=cpu_count)
