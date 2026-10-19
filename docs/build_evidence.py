"""Build the report, test log, and declaration PDFs from the saved artifacts.

fpdf2 is used only here. Training and the API do not import it.

    python docs/build_evidence.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fpdf import FPDF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    REGISTRATION_NUMBER,
    STUDENT_ID,
    STUDENT_NAME,
    VERIFICATION_CODE,
)
from src.inference import predict_record  # noqa: E402
from src.preprocess import NUMERIC_FEATURES  # noqa: E402

COMMIT = "05c06d9a76a95bf3bdb8fde4d824ad8ba50c3825"
FINGERPRINT = "209caa6bff4616bdeee0e6de48554e870d27289974057935d7287f4655323b10"


def load_json(name: str) -> dict:
    """Read one artifact file."""
    path = PROJECT_ROOT / "artifacts" / name
    return json.loads(path.read_text(encoding="utf-8"))


def farm_row(frame: pd.DataFrame, row_id: int) -> pd.Series:
    """Return one saved CSV row. row_id 1 is the first data row."""
    numbered = frame.copy()
    numbered.insert(0, "row_id", range(1, len(numbered) + 1))
    return numbered.loc[numbered["row_id"] == row_id].iloc[0]


def fertilizer_median() -> float:
    """Read the fertilizer median that the fitted yield model imputes."""
    pipeline = joblib.load(PROJECT_ROOT / "models" / "regression_pipeline.joblib")
    imputer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
    position = NUMERIC_FEATURES.index("fertilizer_kg")
    return float(imputer.statistics_[position])


class EvidencePDF(FPDF):
    """Shared page furniture for the submission PDFs."""

    def __init__(self, doc_title: str) -> None:
        super().__init__(format="A4", unit="mm")
        self.doc_title = doc_title
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(14, 14, 14)
        self.set_title(doc_title)
        self.set_author(STUDENT_NAME)

    def header(self) -> None:
        self.set_font("Helvetica", size=8)
        self.set_text_color(60, 60, 60)
        self.cell(0, 5, f"{VERIFICATION_CODE}   {self.doc_title}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(60, 60, 60)
        self.cell(0, 6, f"{STUDENT_NAME}   {REGISTRATION_NUMBER}   page {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0, 0, 0)

    def heading(self, text: str) -> None:
        self.set_font("Helvetica", "B", 14)
        self.multi_cell(0, 7, text, align="L")
        self.ln(1)

    def subhead(self, text: str) -> None:
        self.set_font("Helvetica", "B", 11)
        self.multi_cell(0, 6, text, align="L")
        self.ln(0.4)

    def body(self, text: str) -> None:
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 5, text, align="L")
        self.ln(1.4)

    def bullet(self, text: str) -> None:
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 5, "- " + text, align="L")
        self.ln(0.4)

    def draw_table(self, rows: list[list[str]]) -> None:
        self.set_font("Helvetica", size=8)
        with self.table(
            rows,
            text_align="LEFT",
            line_height=5,
            width=self.epw,
            padding=1,
        ):
            pass
        self.ln(2)

    def caption(self, text: str) -> None:
        self.set_font("Helvetica", "I", 8)
        self.multi_cell(0, 4, text, align="L")
        self.ln(1)


def build_report(prediction: dict, median_fertilizer: float) -> EvidencePDF:
    """Create the seven-page technical report."""
    regression = load_json("regression_metrics.json")
    classification = load_json("classification_metrics.json")
    clustering = load_json("clustering_report.json")
    frame = pd.read_csv(PROJECT_ROOT / "data" / "AI_A2_2527014.csv")
    unique = frame.drop_duplicates()
    median_yph = float((unique["yield_tons"] / unique["field_size_ha"]).median())
    row_241 = farm_row(frame, 241)
    row_41 = farm_row(frame, 41)
    yph_241 = float(row_241["yield_tons"] / row_241["field_size_ha"])
    yph_41 = float(row_41["yield_tons"] / row_41["field_size_ha"])

    pdf = EvidencePDF("Technical report")
    pdf.alias_nb_pages()

    pdf.add_page()
    pdf.heading("1. Identity and decision problem")
    pdf.body(
        f"{STUDENT_NAME}, registration number {REGISTRATION_NUMBER}, "
        f"username {STUDENT_ID}. Verification code {VERIFICATION_CODE}. "
        "Course: SWE 3513 Artificial Intelligence, INES-Ruhengeri. "
        f"System commit {COMMIT} on branch dataset-generation-validation."
    )
    pdf.subhead("Stakeholder and questions")
    pdf.body(
        "The stakeholder is a field officer for a potato cooperative in Musanze. "
        "Farm records arrive in inconsistent spreadsheets, and the officer needs a local "
        "tool that can answer three questions before a visit."
    )
    pdf.bullet("How many tons is this farm likely to harvest? This is the regression task, target yield_tons.")
    pdf.bullet(
        "Is the farm at high production risk? high_risk is 1 when yield per hectare on the "
        "generated base rows is below the median. This is the classification task."
    )
    pdf.bullet("Which farms share similar growing conditions? This is the clustering task.")
    pdf.subhead("Ethical boundary")
    pdf.body(
        "This is a classroom decision-support prototype. The dataset is synthetic. It is "
        "generated from registration number 25/27014 and it is not a record of real Musanze farms. "
        "The service does not replace an agronomist. It must not be used to grant or deny finance, "
        "insurance, subsidies, or any public service. A person reviews every estimate before anyone acts on it."
    )

    pdf.add_page()
    pdf.heading("2. Data and reproducibility")
    pdf.body(
        "LLLL is 7014, the last four digits of 25/27014. The generator calls "
        "numpy.random.default_rng(7014) and rejects any other seed. "
        "Base rows = 240 + (7014 mod 41) = 243. "
        "m = round(0.05 x 243) = 12, so 12 soil-moisture cells and 12 fertilizer cells are cleared. "
        "Three exact duplicate rows are then appended. The saved file has 246 rows. "
        "Numeric values use four decimal places."
    )
    pdf.body(f"SHA-256 of the CSV bytes, also stored in artifacts/data_fingerprint.txt: {FINGERPRINT}")
    pdf.draw_table(
        [
            ["Column", "Rule", "Role"],
            ["rainfall_mm", "Uniform 600 to 1400", "Feature"],
            ["avg_temp_c", "Uniform 12 to 26", "Feature"],
            ["soil_moisture_pct", "Uniform 20 to 80, then 12 missing", "Feature"],
            ["fertilizer_kg", "Uniform 0 to 250, then 12 missing", "Feature"],
            ["field_size_ha", "Uniform 0.2 to 4.0", "Feature"],
            ["season", "A or B", "Feature"],
            ["cooperative_support", "0 or 1", "Feature"],
            ["yield_tons", "Formula below, minimum 0.50", "Regression target"],
            ["high_risk", "1 if yield per hectare is below the median", "Classification target"],
        ]
    )
    pdf.body(
        "yield_tons = 0.004 x rainfall + 0.06 x soil moisture + 0.02 x fertilizer "
        "+ 1.30 x field size - 0.25 x absolute value of (temperature - 18) "
        "+ 0.70 x cooperative support + season effect + noise. "
        "Season A adds 0.50 and season B subtracts 0.20. Noise is normal with mean 0 and "
        "standard deviation 0.80. The result is clipped at 0.50 tons. "
        "high_risk is computed on the complete base rows, before the missing cells are planted."
    )
    pdf.body(
        "Validation checks the column names, the ranges, the 12 missing values on the "
        "deduplicated rows, and the 3 extra copies. Training then drops those 3 copies "
        "before the split, so one farm cannot appear in both sets. Moisture and fertilizer "
        "stay missing until the imputer, and that imputer is fit on the training rows only."
    )

    pdf.add_page()
    pdf.heading("3. Pipeline design")
    pdf.body(
        f"After the copies are removed, {regression['train_rows'] + regression['test_rows']} farms remain. "
        f"train_test_split uses test_size {regression['test_size']}, random_state 7014, and stratifies on high_risk. "
        f"That gives {regression['train_rows']} training farms and {regression['test_rows']} test farms. "
        "Regression, classification, and clustering all use this same split."
    )
    pdf.subhead("Train-only preprocessing")
    pdf.body(
        "Each model is a scikit-learn Pipeline. The first step is a ColumnTransformer. "
        "Numeric columns are rainfall, temperature, soil moisture, fertilizer, field size, and "
        "cooperative support. They pass through a median imputer and then a standard scaler. "
        "Season passes through a most-frequent imputer and then a one-hot encoder. "
        "fit is called on the training frame only. The test frame is transformed with that fit."
    )
    pdf.subhead("Leakage controls")
    pdf.bullet("Inputs are the seven growing-condition columns. yield_tons, yield per hectare, and high_risk are rejected as inputs.")
    pdf.bullet("high_risk is not used to predict yield, and yield is not used to predict high_risk.")
    pdf.bullet("Cluster profiles describe growing conditions. They do not use the targets.")
    pdf.bullet("Duplicate rows are removed before the split, so a copied farm cannot leak into the test set.")
    pdf.subhead("Candidates")
    pdf.bullet("Regression: a mean baseline, linear regression, and a random forest of 200 trees with max depth 6, min_samples_leaf 3, and random_state 7014.")
    pdf.bullet("Classification: a majority-class baseline, logistic regression, and a random forest with the same forest settings. The positive class is high_risk = 1.")
    pdf.bullet("Clustering: KMeans for K = 2, 3, 4, 5, and 6, with n_init 10 and random_state 7014. K is the value with the highest training silhouette.")

    pdf.add_page()
    pdf.heading("4. Regression and classification")
    pdf.subhead("Yield, held-out test set")
    reg_rows = [["Model", "MAE", "RMSE", "R2"]]
    reg_rows.append(
        [
            "mean_baseline",
            f"{regression['baseline']['mae']:.6f}",
            f"{regression['baseline']['rmse']:.6f}",
            f"{regression['baseline']['r2']:.6f}",
        ]
    )
    for candidate in regression["candidates"]:
        reg_rows.append(
            [
                candidate["name"],
                f"{candidate['mae']:.6f}",
                f"{candidate['rmse']:.6f}",
                f"{candidate['r2']:.6f}",
            ]
        )
    pdf.draw_table(reg_rows)
    pdf.body(regression["selection_reason"] + " These figures are the values stored in artifacts/regression_metrics.json.")
    pdf.body(
        "The residual plot shows test errors scattered around zero across the predicted range. "
        "I do not see a separate band of failures at only the small or only the large predictions. "
        "The linear RMSE is 0.842947 tons. The yield formula adds noise with standard deviation 0.80, "
        "so most of the remaining error is that built-in noise, plus the farms whose fertilizer or "
        "moisture had to be imputed. The forest is less accurate here because the designed yield "
        "relationship is linear in the recorded inputs, apart from the absolute temperature term."
    )
    pdf.image(PROJECT_ROOT / "artifacts" / "plots" / "residuals.png", w=118)
    pdf.caption("Held-out residuals for linear regression. Actual minus predicted, in tons.")
    pdf.body(
        f"The largest absolute test error is row 241. Fertilizer is missing, so the model fills it with "
        f"the training median of {median_fertilizer:.1f} kg. The farm has {row_241['field_size_ha']:.4f} ha, "
        f"{row_241['rainfall_mm']:.4f} mm of rainfall, temperature {row_241['avg_temp_c']:.4f} C, season "
        f"{row_241['season']}, and cooperative support {int(row_241['cooperative_support'])}. "
        f"Actual yield is {row_241['yield_tons']:.4f} tons and the prediction is 15.1372 tons, an overestimate "
        f"of 2.2603 tons (artifacts/test_predictions.csv). Temperature is well above 18 C, and the formula "
        f"penalises that gap with an absolute value, which a single straight line only approximates. "
        f"Yield per hectare on the saved values is {yph_241:.4f}, against a base-row median of {median_yph:.4f}. "
        "The risk model still flags this farm, with probability 0.9887. The tonnage is the part a field "
        "officer should not treat as exact while fertilizer is blank."
    )

    pdf.add_page()
    pdf.subhead("High risk, held-out test set")
    cls_rows = [["Model", "Precision", "Recall", "F1", "ROC-AUC"]]
    baseline = classification["baseline"]
    cls_rows.append(
        [
            "majority_baseline",
            f"{baseline['precision']:.6f}",
            f"{baseline['recall']:.6f}",
            f"{baseline['f1']:.6f}",
            f"{baseline['roc_auc']:.6f}",
        ]
    )
    for candidate in classification["candidates"]:
        cls_rows.append(
            [
                candidate["name"],
                f"{candidate['precision']:.6f}",
                f"{candidate['recall']:.6f}",
                f"{candidate['f1']:.6f}",
                f"{candidate['roc_auc']:.6f}",
            ]
        )
    pdf.draw_table(cls_rows)
    pdf.body(classification["selection_reason"] + " These figures are the values stored in artifacts/classification_metrics.json.")
    pdf.body(
        "Rows of the confusion matrix are the actual class and columns are the prediction, "
        "in the order low risk then high risk. Logistic regression has 24 true low-risk farms, "
        "1 false alarm, 1 missed high-risk farm, and 23 true high-risk farms. The majority baseline "
        "predicts low risk for all 49 test farms, so its high-risk precision, recall, and F1 are 0. "
        "The forest makes the same two mistakes and ties logistic regression on F1. Logistic regression "
        "is kept because the ranking of probabilities is slightly better."
    )
    pdf.image(PROJECT_ROOT / "artifacts" / "plots" / "confusion_matrix.png", w=100)
    pdf.caption("Confusion matrix for the selected logistic regression model.")
    pdf.body(
        f"The missed farm is row 41. All of its inputs are present: rainfall {row_41['rainfall_mm']:.4f} mm, "
        f"temperature {row_41['avg_temp_c']:.4f} C, soil moisture {row_41['soil_moisture_pct']:.4f} percent, "
        f"fertilizer {row_41['fertilizer_kg']:.4f} kg, field size {row_41['field_size_ha']:.4f} ha, "
        f"season {row_41['season']}, and cooperative support {int(row_41['cooperative_support'])}. "
        f"Actual yield is {row_41['yield_tons']:.4f} tons and the yield model predicts 11.8744 tons, "
        f"an error of only -0.1382 tons. Yield per hectare is {yph_41:.4f}, and the median on the saved "
        f"base rows is {median_yph:.4f}, so the farm sits just below the cutoff that defines high_risk. "
        "The high-risk probability is 0.3912, on the low-risk side of 0.5. I would not treat this as a "
        "wild miss. It is a borderline farm, and a person should look again when the probability is near one half."
    )

    pdf.add_page()
    pdf.heading("5. Clustering")
    k_rows = [["K", "Silhouette on the training features"]]
    for score in clustering["k_scores"]:
        k_rows.append([str(score["k"]), f"{score['silhouette']:.6f}"])
    pdf.draw_table(k_rows)
    pdf.body(clustering["selection_reason"])
    for profile in clustering["profiles"]:
        pdf.body(profile["profile"])
    pdf.body(
        "Operationally, cluster 0 is the training farms with cooperative support and cluster 1 is "
        "the training farms without it. Rainfall, temperature, moisture, fertilizer, and field size "
        "stay near the training average in both groups, so I would not send different agronomic advice "
        "to the two groups. A field officer can use the label to organise contact with supported and "
        "unsupported farms. The silhouette is only 0.155880, and the display plot overlaps, so the "
        "groups are a rough aid. clusters.png is a two-component PCA of the scaled training features "
        "for display. Membership comes from KMeans on the full scaled feature set. Yield and high_risk "
        "were not used."
    )
    pdf.image(PROJECT_ROOT / "artifacts" / "plots" / "clusters.png", w=120)
    pdf.caption("Training farms coloured by the selected KMeans labels. The axes are for display only.")

    pdf.add_page()
    pdf.heading("6. API, testing, and model card")
    pdf.subhead("Endpoint contract")
    pdf.bullet("GET /health returns status ok and the verification code AI-A2-7014.")
    pdf.bullet("GET /model-info returns the student name, registration number, selected models, K, feature list, and the dataset SHA-256.")
    pdf.bullet("POST /predict accepts the seven features and returns yield_tons, high_risk, high_risk_probability, cluster, and the verification code.")
    pdf.body(
        "samples/valid.json is rainfall 1100, temperature 18.5, soil moisture 55, fertilizer 120, "
        f"field size 1.5, season A, and cooperative support 1. The service returns yield_tons "
        f"{prediction['yield_tons']}, high_risk {prediction['high_risk']}, high_risk_probability "
        f"{prediction['high_risk_probability']}, cluster {prediction['cluster']}, and verification code "
        f"{prediction['verification_code']}."
    )
    pdf.body(
        "samples/invalid.json is the same farm with rainfall 50. Both predict.py and POST /predict "
        "reject it. The API returns HTTP 422 with field rainfall_mm and the message that the value "
        "must be between 600 and 1400. No score is returned. Omitting soil_moisture_pct is also a 422. "
        "Sending season as a number is a 422. The command-line tool exits with status 1 and prints the reason."
    )
    pdf.subhead("Tests")
    pdf.body(
        "pytest -q collects six tests in tests/test_service.py, and all six passed when this report was built. "
        "test_valid_inference_returns_the_verification_code checks the command and the score fields. "
        "test_missing_field_is_rejected, test_wrong_type_is_rejected, and test_range_error_is_rejected "
        "each expect a rejection from the scorer and a 422 from the API. "
        "test_dataset_and_prediction_are_reproducible rebuilds the frame twice with seed 7014, checks the "
        "CSV hash against data_fingerprint.txt, and scores the same farm twice. "
        "test_api_health_model_info_and_predict calls /health, /model-info, and /predict."
    )
    pdf.subhead("Model card")
    pdf.body(
        "Intended use: help a field officer, in a classroom setting, estimate harvest tons, notice farms "
        "the model marks as higher risk, and see which recorded farms share cooperative support. "
        "Prohibited use: credit, insurance, subsidies, public-service decisions, and any claim that the "
        "number is a real agronomic measurement."
    )
    pdf.body(
        "Limits and risks: the file has 243 farms after the copies are removed, and only 49 are held out. "
        "The high F1 reflects a strong designed signal on a small test set, not a survey of Musanze. "
        "A blank fertilizer cell is filled with the training median, which is what went wrong for row 241. "
        "The clusters mostly rediscover a column that is already on the form. A field officer must review "
        "every estimate, and should look harder when an input is missing or the risk probability is near one half."
    )
    pdf.body(
        "With real cooperative records I would replace the formula and the median risk rule with measured "
        "harvests and an agronomist's label, keep a validation split separate from the final test farms, "
        "and check whether cooperative support is a cause or only a flag already written on the sheet."
    )
    return pdf


def build_test_log(quiet_output: str, verbose_output: str) -> EvidencePDF:
    """Save the pytest output that was just captured."""
    pdf = EvidencePDF("Test log")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.heading("Test log")
    pdf.body(
        f"{STUDENT_NAME}, {REGISTRATION_NUMBER}, {VERIFICATION_CODE}. "
        f"Generated {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}. "
        "Python 3.14.4 on Ubuntu 26.04.1 LTS. Working directory: the project root."
    )
    pdf.subhead("Command: pytest -q")
    pdf.set_font("Courier", size=8)
    pdf.multi_cell(0, 4, quiet_output.strip() or "(no output)", align="L")
    pdf.ln(3)
    pdf.subhead("Command: pytest -v --tb=no")
    pdf.set_font("Courier", size=8)
    pdf.multi_cell(0, 4, verbose_output.strip() or "(no output)", align="L")
    pdf.ln(3)
    pdf.set_font("Helvetica", size=10)
    pdf.body(
        "The six tests cover a valid prediction, a missing field, a wrong type, a value outside range, "
        "reproducibility of the dataset and the score, and the health, model-info, and predict responses. "
        "The three invalid cases assert a 422 response, so the log is not limited to the happy path."
    )
    return pdf


def build_declaration() -> EvidencePDF:
    """One-page declaration with a line for a handwritten signature."""
    pdf = EvidencePDF("Declaration")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.heading("Individual declaration")
    pdf.body(f"Name: {STUDENT_NAME}")
    pdf.body(f"Registration number: {REGISTRATION_NUMBER}")
    pdf.body(f"Institutional username: {STUDENT_ID}")
    pdf.body(f"Verification code: {VERIFICATION_CODE}")
    pdf.body("Programme item: SWE 3513 Artificial Intelligence, Individual Assignment 2")
    pdf.body("Date: 18 October 2026")
    pdf.body(
        "I submit the Musanze Harvest Intelligence Service as my individual assignment. "
        "The dataset, fingerprint, metrics, plots, and saved models were produced by running the "
        "code in this repository with seed 7014. The dataset is synthetic. I will not present it "
        "as a record of real farms."
    )
    pdf.body(
        "An AI coding assistant in Cursor helped draft the programs and the report. That help is "
        "described in evidence/AI_USE.md. I am responsible for explaining the system and for making "
        "a small change to it if the assessor asks, such as a range rule or a response field."
    )
    pdf.body(
        "I understand that the service is a classroom prototype. It does not replace an agronomist "
        "and it must not be used to decide finance, insurance, or access to a public service."
    )
    pdf.ln(8)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, "Signature: ________________________________", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Date: 18 October 2026", new_x="LMARGIN", new_y="NEXT")
    return pdf


def run_pytest(args: list[str]) -> str:
    """Run pytest and keep both output streams."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(output)
    return output


def main() -> None:
    """Write the three PDFs next to the project."""
    valid = json.loads((PROJECT_ROOT / "samples" / "valid.json").read_text(encoding="utf-8"))
    prediction = predict_record(valid)
    report = build_report(prediction, fertilizer_median())
    report.output(PROJECT_ROOT / "docs" / "AI_A2_2527014_REPORT.pdf")
    page_count = report.pages_count
    if page_count < 5 or page_count > 7:
        raise RuntimeError(f"Report length must be 5 to 7 pages, got {page_count}.")

    log = build_test_log(run_pytest(["-q"]), run_pytest(["-v", "--tb=no"]))
    evidence = PROJECT_ROOT / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    log.output(evidence / "AI_A2_2527014_TEST_LOG.pdf")
    build_declaration().output(evidence / "AI_A2_2527014_DECLARATION.pdf")
    print(f"Report pages: {page_count}")
    print(f"Valid prediction cited in the report: {prediction}")


if __name__ == "__main__":
    main()
