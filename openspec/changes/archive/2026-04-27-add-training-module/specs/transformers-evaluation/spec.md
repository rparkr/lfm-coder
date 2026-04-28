## ADDED Requirements

### Requirement: Step-based Evaluation Reporting
The evaluation system SHALL support generating results with a `model_id` that includes the current training step to enable correlation between training progress and benchmark performance.

#### Scenario: Evaluation during training
- **WHEN** evaluation is triggered by the training module
- **THEN** the results are logged with a `model_id` like `{base_model}-step-{current_step}`
