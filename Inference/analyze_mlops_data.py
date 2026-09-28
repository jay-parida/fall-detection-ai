
import csv
import os
from collections import defaultdict
from statistics import mean


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\AI_Projects\Fall_Detection_AI"

CSV_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "mlops_logs.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """
    Clean a CSV value before using it.

    This handles older malformed values such as:
        **movement
        **12.5

    Returns a cleaned string.
    """
    if value is None:
        return ""

    value = str(value).strip()

    # Remove accidental leading **
    while value.startswith("**"):
        value = value[2:]

    return value.strip()


def to_float(value):
    """
    Safely convert a value to float.

    Returns None if the value cannot be converted.
    """
    value = clean_value(value)

    if value == "":
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def to_bool(value):
    """
    Convert common boolean representations into True/False.
    """
    value = clean_value(value).lower()

    return value in {
        "true",
        "1",
        "yes",
        "y",
        "on"
    }


def average(values):
    """
    Safely calculate an average.
    """
    valid_values = [
        value for value in values
        if value is not None
    ]

    if not valid_values:
        return None

    return mean(valid_values)


def format_number(value, decimals=2):
    """
    Format numbers for readable terminal output.
    """
    if value is None:
        return "N/A"

    return f"{value:.{decimals}f}"


# ============================================================
# LOAD CSV
# ============================================================

def load_data():
    if not os.path.exists(CSV_PATH):
        print("=" * 70)
        print("ERROR: MLOps CSV file was not found.")
        print("=" * 70)
        print()
        print(f"Expected file:")
        print(CSV_PATH)
        print()
        print("Make sure the backend has generated mlops_logs.csv.")
        return []

    rows = []

    with open(
        CSV_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            cleaned_row = {}

            for key, value in row.items():
                if key is not None:
                    cleaned_row[key.strip()] = clean_value(value)

            rows.append(cleaned_row)

    return rows


# ============================================================
# EVENT START DETECTION
# ============================================================

def count_event_starts(rows):
    """
    Count transitions from event_active=False to True.

    This prevents one fall event from being counted many
    times simply because it remained active for several
    logged samples.
    """

    event_starts = defaultdict(int)

    previous_active = False

    for row in rows:

        active = to_bool(
            row.get("event_active", "")
        )

        lighting = clean_value(
            row.get("lighting", "")
        )

        if active and not previous_active:

            if lighting:
                event_starts[lighting] += 1
            else:
                event_starts["UNKNOWN"] += 1

        previous_active = active

    return event_starts


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze():

    print()
    print("=" * 70)
    print("STAGE 6.5 — MLOps Data Analysis")
    print("=" * 70)
    print()

    rows = load_data()

    if not rows:
        print("No data available for analysis.")
        return

    print(f"CSV file:")
    print(CSV_PATH)
    print()

    print(f"Total logged samples: {len(rows)}")
    print()

    # --------------------------------------------------------
    # Group rows by lighting condition
    # --------------------------------------------------------

    lighting_groups = defaultdict(list)

    for row in rows:

        lighting = clean_value(
            row.get("lighting", "")
        )

        if not lighting:
            lighting = "UNKNOWN"

        lighting_groups[lighting].append(row)

    # --------------------------------------------------------
    # Event starts
    # --------------------------------------------------------

    event_starts = count_event_starts(rows)

    # --------------------------------------------------------
    # Overall counters
    # --------------------------------------------------------

    active_samples = 0
    candidate_samples = 0

    for row in rows:

        if to_bool(row.get("event_active", "")):
            active_samples += 1

        if to_bool(row.get("candidate", "")):
            candidate_samples += 1

    active_percentage = (
        active_samples / len(rows) * 100
        if rows
        else 0
    )

    candidate_percentage = (
        candidate_samples / len(rows) * 100
        if rows
        else 0
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    print("-" * 70)
    print("OVERALL SUMMARY")
    print("-" * 70)

    print(f"Total samples              : {len(rows)}")
    print(f"Event-active samples       : {active_samples}")
    print(f"Event-active percentage    : {active_percentage:.2f}%")
    print(f"Candidate samples          : {candidate_samples}")
    print(f"Candidate percentage       : {candidate_percentage:.2f}%")
    print()

    # ========================================================
    # LIGHTING SUMMARY
    # ========================================================

    print("-" * 70)
    print("LIGHTING CONDITION SUMMARY")
    print("-" * 70)

    lighting_order = [
        "VERY_DARK",
        "DARK",
        "NORMAL",
        "BRIGHT",
        "VERY_BRIGHT",
        "UNKNOWN"
    ]

    # Print known lighting categories first
    ordered_conditions = []

    for condition in lighting_order:
        if condition in lighting_groups:
            ordered_conditions.append(condition)

    # Add any unexpected categories afterward
    for condition in sorted(lighting_groups.keys()):
        if condition not in ordered_conditions:
            ordered_conditions.append(condition)

    for lighting in ordered_conditions:

        group = lighting_groups[lighting]

        brightness_values = [
            to_float(row.get("brightness"))
            for row in group
        ]

        contrast_values = [
            to_float(row.get("contrast"))
            for row in group
        ]

        blur_values = [
            to_float(row.get("blur_score"))
            for row in group
        ]

        confidence_values = [
            to_float(row.get("confidence"))
            for row in group
        ]

        group_active_samples = sum(
            to_bool(row.get("event_active"))
            for row in group
        )

        group_candidate_samples = sum(
            to_bool(row.get("candidate"))
            for row in group
        )

        group_active_percentage = (
            group_active_samples / len(group) * 100
            if group
            else 0
        )

        print()
        print(f"[{lighting}]")

        print(
            f"  Samples                  : {len(group)}"
        )

        print(
            f"  Average brightness       : "
            f"{format_number(average(brightness_values))}"
        )

        print(
            f"  Average contrast         : "
            f"{format_number(average(contrast_values))}"
        )

        print(
            f"  Average blur score       : "
            f"{format_number(average(blur_values))}"
        )

        print(
            f"  Average confidence       : "
            f"{format_number(average(confidence_values))}"
        )

        print(
            f"  Fall event starts        : "
            f"{event_starts.get(lighting, 0)}"
        )

        print(
            f"  Candidate samples        : "
            f"{group_candidate_samples}"
        )

        print(
            f"  Event-active samples     : "
            f"{group_active_samples}"
        )

        print(
            f"  Event-active percentage  : "
            f"{group_active_percentage:.2f}%"
        )

    # ========================================================
    # MODEL BEHAVIOR OBSERVATIONS
    # ========================================================

    print()
    print("-" * 70)
    print("MODEL BEHAVIOR OBSERVATIONS")
    print("-" * 70)

    print()
    print(
        "These observations describe the collected monitoring data."
    )

    print(
        "They are NOT measurements of model accuracy because "
        "ground-truth labels are not present in this CSV."
    )

    print()

    # Average confidence comparison
    condition_confidences = {}

    for lighting in ordered_conditions:

        group = lighting_groups[lighting]

        confidence_values = [
            to_float(row.get("confidence"))
            for row in group
        ]

        avg_confidence = average(confidence_values)

        if avg_confidence is not None:
            condition_confidences[lighting] = avg_confidence

    if condition_confidences:

        print(
            "Average confidence by lighting condition:"
        )

        for lighting in ordered_conditions:

            if lighting in condition_confidences:

                print(
                    f"  {lighting:<12} "
                    f"{condition_confidences[lighting]:.3f}"
                )

        print()

    # Event distribution
    total_events = sum(event_starts.values())

    print(
        f"Total detected fall-event starts in the logs: "
        f"{total_events}"
    )

    if total_events > 0:

        print()
        print("Fall-event starts by lighting condition:")

        for lighting in ordered_conditions:

            count = event_starts.get(lighting, 0)

            if count > 0:

                percentage = (
                    count / total_events * 100
                )

                print(
                    f"  {lighting:<12} "
                    f"{count} event(s) "
                    f"({percentage:.2f}% of logged event starts)"
                )

    else:

        print()
        print(
            "No fall-event starts were recorded in this dataset."
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    print()
    print("-" * 70)
    print("DATA QUALITY CHECK")
    print("-" * 70)

    required_fields = [
        "timestamp",
        "lighting",
        "brightness",
        "contrast",
        "blur_score",
        "status",
        "event_active",
        "confidence",
        "movement",
        "vertical",
        "direction",
        "ratio",
        "ratio_change",
        "recovery",
        "candidate",
        "score",
        "event_timer"
    ]

    if rows:

        available_fields = set(rows[0].keys())

        missing_fields = [
            field
            for field in required_fields
            if field not in available_fields
        ]

        if missing_fields:

            print(
                "Missing expected fields:"
            )

            for field in missing_fields:
                print(f"  - {field}")

        else:

            print(
                "All expected logging fields are present."
            )

    # Count rows with invalid numeric values
    numeric_fields = [
        "brightness",
        "contrast",
        "blur_score",
        "confidence",
        "movement",
        "vertical",
        "ratio",
        "ratio_change",
        "score",
        "event_timer"
    ]

    invalid_counts = defaultdict(int)

    for row in rows:

        for field in numeric_fields:

            value = row.get(field, "")

            if value == "":
                invalid_counts[field] += 1
                continue

            if to_float(value) is None:
                invalid_counts[field] += 1

    invalid_fields = {
        field: count
        for field, count in invalid_counts.items()
        if count > 0
    }

    if invalid_fields:

        print()
        print("Rows with missing/invalid numeric values:")

        for field, count in invalid_fields.items():

            print(
                f"  {field:<15} : {count}"
            )

    else:

        print(
            "No missing/invalid numeric values detected."
        )

    # ========================================================
    # FINAL NOTE
    # ========================================================

    print()
    print("=" * 70)
    print("STAGE 6.5 ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        "Next step will be decided from the actual collected data."
    )

    print(
        "The analysis does not modify the model, detector, or CSV."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    analyze()
