# AI use disclosure

**Student:** Mohammed Osama Hasan  
**Registration number:** 25/27014  
**Verification code:** AI-A2-7014  
**Date:** 18 October 2026

An AI coding assistant in Cursor was used while building this assignment. This note says what that help covered, and what still has to come from running the code.

## What the assistant did

- Drafted the Python modules for data generation, validation, preprocessing, the three models, the command-line predictor, the FastAPI service, and the tests.
- Drafted this disclosure, the technical report, the test log, and the declaration from the assignment brief and from the artifacts produced on this machine.
- Used the registration number 25/27014, which fixes seed 7014, the row count, and the verification code `AI-A2-7014`.

## What was not invented

- The CSV, the SHA-256 fingerprint, the metrics, the plots, and the saved models come from running `run_all.py --student-id ug2527014@ines.ac.rw --seed 7014`.
- The yield error for row 241 and the missed high-risk farm in row 41 are rows in `artifacts/test_predictions.csv`. The report describes those rows. It does not replace them with a generic example.
- The cluster profiles describe the training farms after KMeans. They are the profiles stored in `artifacts/clustering_report.json`.

## Responsibility

I have to be able to explain the seed, the train and test split, why linear regression and logistic regression were selected, why row 241 is a bad yield estimate, why row 41 is a missed high-risk farm, and why the two clusters mainly separate cooperative support. I also have to be able to change the code during the defense, for example a range check or a response field.

The dataset is synthetic. It must not be described as a survey of real Musanze farms. The service is a classroom prototype. It does not replace an agronomist and it must not decide access to finance, insurance, or public services.
