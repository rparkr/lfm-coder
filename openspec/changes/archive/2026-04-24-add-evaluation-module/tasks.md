## 1. Core Data Structures

- [x] 1.1 Create evaluation result dataclasses in `src/lfm_coder/evals/types.py`
- [x] 1.2 Define `EvaluationResult`, `TaskResult`, `DatasetMetrics` classes

## 2. Base Evaluator Class

- [x] 2.1 Create base `Evaluator` class in `src/lfm_coder/evals/eval.py`
- [x] 2.2 Implement code extraction logic (last fenced code block)
- [x] 2.3 Implement metric calculation (pass rate, format success, etc.)
- [x] 2.4 Implement checkpoint/resume logic with JSONL
- [x] 2.5 Add tqdm progress bars
- [x] 2.6 Add logging throughout

## 3. Transformers Evaluator

- [x] 3.1 Create `TransformersEvaluator` class extending base Evaluator
- [x] 3.2 Implement `generate` method using transformers model
- [x] 3.3 Add temperature and batch size configuration
- [x] 3.4 Add token counting

## 4. OpenAI-Compatible Evaluator

- [x] 4.1 Create `OpenAICompatibleEvaluator` class in `src/lfm_coder/evals/openai_evaluator.py`
- [x] 4.2 Implement `generate` method using httpx client
- [x] 4.3 Add environment variable configuration for base_url and api_key
- [x] 4.4 Handle API errors gracefully

## 5. Integration with Datasets

- [x] 5.1 Integrate with HumanEvalPlusDataset
- [x] 5.2 Integrate with MBPPPlusDataset (long tests only)
- [x] 5.3 Add per-task accuracy tracking

## 6. Checkpointing and Metrics

- [x] 6.1 Implement JSONL file saving with timestamp
- [x] 6.2 Implement resume from checkpoint logic
- [x] 6.3 Add trackio integration for metrics (Note: Metrics are returned to be logged externally as per user feedback)
- [x] 6.4 Add comprehensive logging

## 7. Testing (using the Pytest framework)

- [x] 7.1 Add unit tests for code extraction
- [x] 7.2 Add unit tests for metric calculation
- [x] 7.3 Add integration tests for evaluator classes
- [x] 7.4 Add tests for checkpoint/resume functionality

## 8. Documentation

- [x] 8.1 Add docstrings to all public classes and methods
- [x] 8.2 Add usage examples to module docstring