## ADDED Requirements

### Requirement: OpenAI-Compatible API Evaluation
The system SHALL evaluate any OpenAI-compatible API endpoint (e.g., Ollama) using the same evaluation logic as transformers models.

#### Scenario: Evaluate model via OpenAI-compatible API
- **WHEN** the evaluator is initialized with OPENAI_BASE_URL and OPENAI_API_KEY environment variables
- **THEN** the system makes API calls to obtain model outputs and evaluates them

#### Scenario: Use custom API endpoint
- **WHEN** the evaluator is initialized with a custom base_url
- **THEN** the system uses that endpoint for all API calls

#### Scenario: Handle API errors gracefully
- **WHEN** the API call fails
- **THEN** the system logs the error, marks the sample as failed, and continues evaluation

#### Scenario: Share evaluation logic with transformers evaluator
- **WHEN** evaluation is performed
- **THEN** the system uses the same code extraction, sandbox execution, and metric calculation as the transformers evaluator