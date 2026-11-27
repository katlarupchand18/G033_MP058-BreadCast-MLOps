# BreadCast API Testing and Validation

## 1. Objective

The objective is to verify the functionality, reliability and input validation of the BreadCast FastAPI prediction service.

## 2. API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks whether the API service is responding. |
| `/predict` | POST | Accepts eight input features and returns the predicted number of loaves sold. |

## 3. Testing Methodology

The API was tested using pytest and FastAPI TestClient. This allows endpoint testing without manually starting the API server.

## 4. Existing Test Cases

| Test scenario | Expected result | Actual result |
|---|---|---|
| Health endpoint | HTTP 200, status ok | Passed |
| Health response structure | Valid status field | Passed |
| Valid prediction input | HTTP 200, non-negative numeric prediction | Passed |
| All valid day types | HTTP 200 | Passed |
| All valid outlets | HTTP 200 | Passed |
| Shelf hours above 30 | HTTP 422 | Passed |
| Previous-day orders below 5 | HTTP 422 | Passed |
| Missing required fields | HTTP 422 | Passed |
| Unexpected fields | HTTP 422 required by API schema | Passed — HTTP 422 confirmed |
| Shelf hours equal to 30 | HTTP 200 | Passed |
| Previous-day orders equal to 5 | HTTP 200 | Passed |

## 5. Additional Edge-Case Testing

Ten additional test cases were introduced to evaluate invalid inputs and boundary conditions.

| Test scenario | Expected response | Actual result |
|---|---|---|
| Invalid day type | 422 | Passed |
| Invalid outlet | 422 | Passed |
| Negative footfall index | 422 | Passed |
| Negative competitor count | 422 | Passed |
| Zero shelf hours | 422 | Passed |
| Negative shelf hours | 422 | Passed |
| Invalid price data type | 422 | Passed |
| Invalid temperature data type | 422 | Passed |
| Zero footfall index | 200 | Passed |
| Zero competitor count | 200 | Passed |

## 6. Test Execution

Command:

`python -m pytest tests/test_api.py tests/test_api_edge_cases.py -v`

Testing environment:

- Operating System: macOS
- Python: 3.12.8
- Pytest: 9.2.0
- Testing Date: 27 November 2026

## 7. Test Results

| Metric | Result |
|---|---|
| Original test cases | 11 |
| Additional test cases | 10 |
| Total test cases | 21 |
| Passed | 21 |
| Failed | 0 |
| Warnings | 1 |
| Pass rate | 100% |
| Execution time | 1.38 seconds |

## 8. Observations

A StarletteDeprecationWarning related to the HTTPX testing dependency was observed. It did not affect successful test execution.

The number of API test cases increased from 11 to 21, representing a 90.9% increase in test count.

## 9. Conclusion

All 21 API tests passed successfully. The tested endpoints correctly handled valid requests and the tested invalid inputs and boundary conditions.

## 10. Contribution Scope

- Execution and review of existing API tests.
- Review of API input validation rules.
- Extension of API edge-case testing.
- Documentation of test execution and results.

No changes to the API implementation, original datasets or trained models were required for these tests.

## 11. Complete Project Test Suite Validation

Command: `python -m pytest tests/ -v`

- Total tests: 42
- Passed: 41
- Failed: 0
- Skipped: 1
- Warnings: 1

All executed tests passed successfully.