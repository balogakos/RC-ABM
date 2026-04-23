# RC-ABM Test Suite

This directory contains a suite of automated tests for the Retail Centre Agent-Based Model. These tests ensure the mathematical correctness of agent behavior, destination choice logic, and simulation stability without requiring the full production datasets.

## Test Files Overview

### 1. `test_agent_logic.py`
**Scope:** Core agent lifecycle and state transitions.
- **Stock Consumption:** Verifies that stock depletes correctly based on consumption rates.
- **Threshold Triggers:** Ensures grocery trips are only triggered when stock falls below the `REORDER_THRESHOLD`.
- **NTS Triggering:** Validates that non-grocery trips fire according to their assigned daily probabilities.

### 2. `test_destination_choice.py`
**Scope:** The spatial choice engine and filtering logic.
- **Amenity Filters:** Confirms that trips (e.g., Comparison) are strictly constrained to retail centres that possess the required amenities (e.g., `Retail = 1`).
- **Softmax Sampling:** Statistically verifies that agents choose higher-utility destinations more frequently according to the Softmax probability distribution.

### 3. `test_model_mechanics.py`
**Scope:** Emerging behavior and model dynamics.
- **Visit Feedback:** Tests the stochastic reinforcement of utility scores (+5% / -5%) after a successful visit.
- **Spatial Diffusion:** Verifies that "Word-of-Mouth" effects correctly boost the attractiveness of popular centres for neighbors in the same postcode.

### 4. `test_integration.py`
**Scope:** Full pipeline stability.
- **End-to-End Run:** Executes a mini-simulation (e.g., 500 agents over 7 days) using synthetic data to ensure no runtime errors occur during the main loop.

---

## How to Run Tests

You can run individual test files directly:

```bash
python tests/test_agent_logic.py
```

Or run the full suite using your preferred test runner:

```bash
python -m unittest discover tests
```

## Creating New Tests
When adding new functionality to `agent.py`, please add a corresponding test case in the relevant file. Use the helper functions in `tests/tests.py` to generate synthetic agent and utility data so that tests remain fast and independent of external files.
