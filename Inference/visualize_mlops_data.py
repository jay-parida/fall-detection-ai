
import csv
import os
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\AI_Projects\Fall_Detection_AI"

CSV_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "mlops_logs.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Data",
    "mlops_visualizations"
)


# Lighting conditions used for the main analysis.
# UNKNOWN is intentionally excluded from the charts because
# those rows contain zero-valued lighting metrics.
LIGHTING_ORDER = [
    "VERY_DARK",
    "DARK",
    "NORMAL",
    "BRIGHT",
    "VERY_BRIGHT"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """
    Clean CSV values.

    Handles accidental leading '**' from older CSV data.
    Example:
        **0.81 -> 0.81
    """

    if value is None:
        return ""

    value = str(value).strip()

    while value.startswith("**"):
        value = value[2:]

    return value.strip()


def to_float(value):
    """
    Safely convert a value to float.
    Returns None if conversion fails.
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
    Convert common boolean representations to True/False.
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
    Safely calculate the average of valid numeric values.
    """

    valid_values = [
        value
        for value in values
        if value is not None
    ]

    if not valid_values:
        return 0.0

    return mean(valid_values)


def save_chart(filename):
    """
    Save the current Matplotlib figure.
    """

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Created: {path}")


# ============================================================
# LOAD CSV
# ============================================================

def load_data():

    if not os.path.exists(CSV_PATH):

        print()
        print("=" * 70)
        print("ERROR: MLOps CSV file was not found.")
        print("=" * 70)
        print()
        print(f"Expected file:")
        print(CSV_PATH)
        print()

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

                    cleaned_row[key.strip()] = clean_value(
                        value
                    )

            rows.append(cleaned_row)

    return rows


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_groups(rows):

    groups = defaultdict(list)

    for row in rows:

        lighting = clean_value(
            row.get("lighting", "")
        )

        if lighting in LIGHTING_ORDER:

            groups[lighting].append(row)

    return groups


# ============================================================
# CHART 1
# SAMPLE COUNT BY LIGHTING
# ============================================================

def chart_sample_count(groups):

    conditions = [
        lighting
        for lighting in LIGHTING_ORDER
        if lighting in groups
    ]

    counts = [
        len(groups[lighting])
        for lighting in conditions
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        conditions,
        counts
    )

    plt.title(
        "Logged Samples by Lighting Condition"
    )

    plt.xlabel(
        "Lighting Condition"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.xticks(
        rotation=20
    )

    save_chart(
        "01_samples_by_lighting.png"
    )


# ============================================================
# CHART 2
# AVERAGE BRIGHTNESS
# ============================================================

def chart_average_brightness(groups):

    conditions = [
        lighting
        for lighting in LIGHTING_ORDER
        if lighting in groups
    ]

    values = []

    for lighting in conditions:

        brightness_values = [
            to_float(row.get("brightness"))
            for row in groups[lighting]
        ]

        values.append(
            average(brightness_values)
        )

    plt.figure(figsize=(10, 6))

    plt.bar(
        conditions,
        values
    )

    plt.title(
        "Average Brightness by Lighting Condition"
    )

    plt.xlabel(
        "Lighting Condition"
    )

    plt.ylabel(
        "Average Brightness"
    )

    plt.xticks(
        rotation=20
    )

    save_chart(
        "02_average_brightness.png"
    )


# ============================================================
# CHART 3
# AVERAGE CONTRAST
# ============================================================

def chart_average_contrast(groups):

    conditions = [
        lighting
        for lighting in LIGHTING_ORDER
        if lighting in groups
    ]

    values = []

    for lighting in conditions:

        contrast_values = [
            to_float(row.get("contrast"))
            for row in groups[lighting]
        ]

        values.append(
            average(contrast_values)
        )

    plt.figure(figsize=(10, 6))

    plt.bar(
        conditions,
        values
    )

    plt.title(
        "Average Contrast by Lighting Condition"
    )

    plt.xlabel(
        "Lighting Condition"
    )

    plt.ylabel(
        "Average Contrast"
    )

    plt.xticks(
        rotation=20
    )

    save_chart(
        "03_average_contrast.png"
    )


# ============================================================
# CHART 4
# AVERAGE BLUR SCORE
# ============================================================

def chart_average_blur(groups):

    conditions = [
        lighting
        for lighting in LIGHTING_ORDER
        if lighting in groups
    ]

    values = []

    for lighting in conditions:

        blur_values = [
            to_float(row.get("blur_score"))
            for row in groups[lighting]
        ]

        values.append(
            average(blur_values)
        )

    plt.figure(figsize=(10, 6))

    plt.bar(
        conditions,
        values
    )

    plt.title(
        "Average Blur Score by Lighting Condition"
    )

    plt.xlabel(
        "Lighting Condition"
    )

    plt.ylabel(
        "Average Blur Score"
    )

    plt.xticks(
        rotation=20
    )

    save_chart(
        "04_average_blur_score.png"
    )


# ============================================================
# CHART 5
# AVERAGE MODEL CONFIDENCE
# ============================================================

def chart_average_confidence(groups):

    conditions = [
        lighting
        for lighting in LIGHTING_ORDER
        if lighting in groups
    ]

    values = []

    for lighting in conditions:

        confidence_values = [
            to_float(row.get("confidence"))
            for row in groups[lighting]
        ]

        values.append(
            average(confidence_values)
        )

    plt.figure(figsize=(10, 6))

    plt.bar(
        conditions,
        values
    )

    plt.title(
        "Average Model Confidence by Lighting Condition"
    )

    plt.xlabel(
        "Lighting Condition"
    )

    plt.ylabel(
        "Average Confidence"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=20
    )

    save_chart(
        "05_average_model_confidence.png"
    )


# ============================================================
# CHART 6
# FALL EVENT STARTS
# ============================================================

def count_event_starts(rows):

    event_starts = defaultdict(int)

    previous_active = False

    for row in rows:

        active = to_bool(
            row.get("event_active")
        )

        lighting = clean_value(
            row.get("lighting", "")
        )

        # Count only the transition:
        #
        # False -> True
        #
        # This prevents one fall event from being counted
        # multiple times across several logging samples.

        if active and not previous_active:

            if lighting in LIGHTING_ORDER:

                event_starts[lighting] += 1

        previous_active = active

    return event_starts


def chart_event_starts(groups, rows):

    event_starts = count_event_starts(rows)

    conditions = [
        lighting
        for lighting in LIGHTING_ORDER
        if lighting in groups
    ]

    values = [
        event_starts.get(lighting, 0)
        for lighting in conditions
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        conditions,
        values
    )

    plt.title(
        "Fall-Event Starts by Lighting Condition"
    )

    plt.xlabel(
        "Lighting Condition"
    )

    plt.ylabel(
        "Number of Event Starts"
    )

    plt.xticks(
        rotation=20
    )

    save_chart(
        "06_fall_event_starts.png"
    )


# ============================================================
# CHART 7
# EVENT-ACTIVE PERCENTAGE
# ============================================================

def chart_event_active_percentage(groups):

    conditions = [
        lighting
        for lighting in LIGHTING_ORDER
        if lighting in groups
    ]

    percentages = []

    for lighting in conditions:

        group = groups[lighting]

        active_count = sum(
            to_bool(row.get("event_active"))
            for row in group
        )

        percentage = (
            active_count / len(group) * 100
            if group
            else 0
        )

        percentages.append(
            percentage
        )

    plt.figure(figsize=(10, 6))

    plt.bar(
        conditions,
        percentages
    )

    plt.title(
        "Event-Active Percentage by Lighting Condition"
    )

    plt.xlabel(
        "Lighting Condition"
    )

    plt.ylabel(
        "Event-Active Samples (%)"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        rotation=20
    )

    save_chart(
        "07_event_active_percentage.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STAGE 6.6 — MLOps Visualization")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        f"Input CSV:"
    )

    print(
        CSV_PATH
    )

    print()

    print(
        f"Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    rows = load_data()

    if not rows:

        print(
            "No data available for visualization."
        )

        return

    print(
        f"Loaded {len(rows)} logged samples."
    )

    print()

    # --------------------------------------------------------
    # Group data by lighting condition
    # --------------------------------------------------------

    groups = prepare_groups(rows)

    print(
        "Lighting conditions found:"
    )

    for lighting in LIGHTING_ORDER:

        count = len(
            groups.get(lighting, [])
        )

        if count > 0:

            print(
                f"  {lighting:<12}: {count} samples"
            )

    unknown_count = sum(
        1
        for row in rows
        if clean_value(
            row.get("lighting", "")
        ) not in LIGHTING_ORDER
    )

    if unknown_count > 0:

        print(
            f"  {'UNKNOWN':<12}: "
            f"{unknown_count} samples "
            f"(excluded from lighting charts)"
        )

    print()

    # --------------------------------------------------------
    # Generate charts
    # --------------------------------------------------------

    print(
        "Generating visualizations..."
    )

    print()

    chart_sample_count(
        groups
    )

    chart_average_brightness(
        groups
    )

    chart_average_contrast(
        groups
    )

    chart_average_blur(
        groups
    )

    chart_average_confidence(
        groups
    )

    chart_event_starts(
        groups,
        rows
    )

    chart_event_active_percentage(
        groups
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("STAGE 6.6 VISUALIZATION COMPLETE")
    print("=" * 70)

    print()

    print(
        "Charts saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Generated charts:"
    )

    print(
        "  01_samples_by_lighting.png"
    )

    print(
        "  02_average_brightness.png"
    )

    print(
        "  03_average_contrast.png"
    )

    print(
        "  04_average_blur_score.png"
    )

    print(
        "  05_average_model_confidence.png"
    )

    print(
        "  06_fall_event_starts.png"
    )

    print(
        "  07_event_active_percentage.png"
    )

    print()

    print(
        "Note: These charts describe collected model behavior."
    )

    print(
        "They do not represent model accuracy because "
        "ground-truth labels are not present."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

