## ADDED Requirements

### Requirement: Evaluation Checkpointing
The system SHALL save and resume evaluation progress to enable recovery from failures.

#### Scenario: Save results to JSONL
- **WHEN** evaluation completes a batch
- **THEN** the system appends results to a JSONL file with timestamp in filename

#### Scenario: Resume from checkpoint
- **WHEN** evaluation is restarted with existing results file
- **THEN** the system reads the file, finds the last completed task, and continues from the next one

#### Scenario: Track completed tasks
- **WHEN** results are saved
- **THEN** each result includes task_id, dataset_name, and completion status

#### Scenario: Use trackio for metrics
- **WHEN** evaluation completes
- **THEN** the system logs metrics using trackio for external tracking systems