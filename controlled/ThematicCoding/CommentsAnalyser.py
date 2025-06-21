import re
import pandas as pd
from pathlib import Path
from pandas.api.types import CategoricalDtype  # Import CategoricalDtype

# Define the paths and scenarios for all files
file_data = {
    'OC': 'original_code_reasoning.txt',
    'I1': 'nocomments_code_reasoning.txt',
    'I2': 'badnames_code_reasoning.txt',
    'I3': 'cleancode_code_reasoning.txt'
}

aspect_keywords = {
    "Poor Variable/Method Names": [
        "poor variable",
        "poor method name",
        "bad variable",
        "bad method name",
        "inconsistent naming",
        "non-standard method name",
        "magic number",
        "magic string",
        "are unclear",
        "unconventional",
        "confusing",
        "cryptic name",
        "cryptic variable",
        "single-letter names",
        "food-related names",
        "vegetable-themed variable names",
        "spice-related name",
        "culinary term",
        "nonsensical name",
        "arbitrary naming"
    ],
    "Poor Readability": [
        "poor readability",
        "hard to read",
        "difficult to understand",
        "unclear",
        "reduce readability",
        "less readable",
        "hinder readability",
        "verbosity",
        "verbose",
        "hard to follow",
        "cognitive load",
        "obscured logic",
        "confusing logic",
        "difficult to comprehend"
    ],
    "Good Code Structure": [
        "well-structured",
        "good structure",
        "well-organized",
        "good separation of concerns",
        "modular design",
        "clean structure",
        "logical organization",
        "well-organized inner classes",
        "well-defined inner classes",
        "simple class structure",
        "clear class structure"
    ],
    "Good Documentation": [
        "comprehensive documentation",
        "good documentation",
        "well-documented",
        "JavaDoc comment",
        "clear comment",
        "descriptive JavaDoc comments",
        "detailed comments",
        "comprehensive Javadoc",
        "clear method comment",
        "detailed JavaDoc"
    ],
    "Lack of Documentation/Comments": [
        "lacks comprehensive comments",
        "lacks comprehensive documentation",
        "lacks documentation",
        "minimal comments",
        "unclear purpose",
        "no documentation",
        "missing javadoc",
        "undocumented",
        "lacks meaningful comments",
        "lack of clear comments"
    ],
    "High Code Complexity": [
        "complex",
        "complexity",
        "nested",
        "over-engineering",
        "overly complex",
        "convoluted",
        "dense logic",
        "excessive nesting",
        "deeply nested",
        "nested conditional logic",
        "high cognitive complexity",
        "intricate structure"
    ],
    "Bad Code Structure": [
        "lacks ... structure",
        "bad separation of concerns",
        "over-engineering",
        "spaghetti code",
        "monolithic",
        "tight coupling",
        "hardcoded data",
        "long methods",
        "class too long",
        "mixed responsibilities",
        "violates single responsibility",
        "poor encapsulation",
        "redundant code",
        "code duplication",
        "complex nested data structures",
        "inconsistent structure"
    ],
    "Adherence to Best Practices": [
        "best practices",
        "Java conventions",
        "object-oriented principles",
        "good practices",
        "Java idioms",
        "follows standard Java conventions",
        "strong adherence to best practices",
        "strong design principles",
        "type safety",
        "design patterns"
    ],
    "Good Implementation/Logic": [
        "excellent implementation",
        "good implementation",
        "clear method implementations",
        "thoughtful design",
        "clean implementation",
        "straightforward",
        "logical",
        "efficient",
        "advanced algorithm",
        "algorithmic thinking",
        "Kahan summation",
        "sound logic",
        "technically sound"
    ],
    "Consistent Formatting": [
        "consistent formatting",
        "well-formatted",
        "consistent styling",
        "consistent indentation",
        "proper implementation"
    ],
    "Good Variable/Method Names": [
        "clear method names",
        "meaningful variable names",
        "descriptive ... names",
        "consistent naming",
        "descriptive method names"
    ],
    "Good Error Handling": [
        "robust error handling",
        "proper exception handling",
        "handles potential exceptions",
        "comprehensive error handling"
    ],
    "Poor Error Handling": [
        "error handling is minimal",
        "swallowed exception",
        "suppressed empty catch block",
        "inconsistent ... error handling",
        "ignores exceptions",
        "empty catch block",
        "lacks input validation",
        "minimal error handling",
        "poor error handling"
    ],
    "Good Readability": [
        "high readability",
        "good readability",
        "clear",
        "easy to understand",
        "readable",
        "self-explanatory",
        "easy to follow"
    ],
    "Unprofessional Comments": [
        "unprofessional"
    ],
    "Inconsistent Formatting": [
        "inconsistent line breaks",
        "inconsistent formatting",
        "inconsistent indentation",
        "messy"
    ],
    "Good Use of Language Features": [
        "modern Java features",
        "switch expressions",
        "lambda",
        "streams",
        "Java generics",
        "temporal interfaces",
        "method overriding"
    ]
}

negative_to_positive_pairs = {
    "Poor Variable/Method Names": "Good Variable/Method Names",
    "Poor Readability": "Good Readability",
    "Bad Code Structure": "Good Code Structure",
    "Lack of Documentation/Comments": "Good Documentation",
    "Poor Error Handling": "Good Error Handling",
    "Inconsistent Formatting": "Consistent Formatting",
}

# Dictionary to store counts for all scenarios
all_scenario_counts = {aspect: {scenario_key: 0 for scenario_key in file_data.keys()} for aspect in aspect_keywords.keys()}

# Process each file
for scenario_key, file_name in file_data.items():
    file_path = Path(file_name)
    current_aspect_counts = {aspect: 0 for aspect in aspect_keywords.keys()}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_content = re.sub(r'\\', '', line).strip().lower()

                if not line_content:
                    continue

                matched_aspects_in_line = set()
                for aspect, keywords in aspect_keywords.items():
                    for keyword in keywords:
                        if '...' in keyword:
                            pattern = r'\b' + re.escape(keyword).replace(r'\.\.\.', r'\s*\w*\s*') + r'\b'
                        else:
                            pattern = r'\b' + re.escape(keyword) + r'\b'

                        if re.search(pattern, line_content):
                            matched_aspects_in_line.add(aspect)
                            break

                # Logic to exclude positive counts if a negative counterpart is present (to prevent false positive)
                final_aspects_to_count = set(matched_aspects_in_line)

                for negative_aspect, positive_aspect in negative_to_positive_pairs.items():
                    # If both the negative and its corresponding positive aspect are in the line
                    if negative_aspect in matched_aspects_in_line and positive_aspect in matched_aspects_in_line:
                        final_aspects_to_count.discard(positive_aspect)

                for aspect in final_aspects_to_count:
                    current_aspect_counts[aspect] += 1

        # Transfer counts from the current file to the overall dictionary
        for aspect, count in current_aspect_counts.items():
            all_scenario_counts[aspect][scenario_key] = count

    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}. Skipping this scenario.")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

# Create DataFrame from the collected data
df_data = []
for aspect, counts_by_scenario in all_scenario_counts.items():
    row = {'Aspect': aspect}
    row.update(counts_by_scenario)
    df_data.append(row)

df = pd.DataFrame(df_data)

# --- NEW: Define the custom order for 'Aspect' column ---
custom_aspect_order = [
    "Adherence to Best Practices",
    "Good Implementation/Logic",
    "Good Use of Language Features",
    "High Code Complexity",
    "Consistent Formatting",
    "Inconsistent Formatting",
    "Good Code Structure",
    "Bad Code Structure",
    "Good Documentation",
    "Lack of Documentation/Comments",
    "Unprofessional Comments",  # Ensure this is in your aspect_keywords if you want it to appear
    "Good Error Handling",
    "Poor Error Handling",
    "Good Readability",
    "Poor Readability",
    "Good Variable/Method Names",
    "Poor Variable/Method Names"
]

# Convert 'Aspect' column to a CategoricalDtype with the specified order
# Any aspects found in data but not in custom_aspect_order will appear at the end, sorted alphabetically.
# Any aspects in custom_aspect_order but not in data will not appear.
df['Aspect'] = pd.Categorical(df['Aspect'], categories=custom_aspect_order, ordered=True)

# Sort the DataFrame by the 'Aspect' column
df_sorted = df.sort_values(by='Aspect').reset_index(drop=True)

# Generate LaTeX table
latex_table = """
\\begin{table}[hbt]
    \\centering
    \\scriptsize
    \\caption{Code Evaluation Across Scenarios}
    \\label{tab:thematic_coding}
    \\begin{tabular}{l""" + "r" * len(file_data) + """}
        \\toprule
        \\textbf{Theme} & """ + " & ".join([f"\\textbf{{{s}}}" for s in file_data.keys()]) + """ \\\\
        \\midrule
"""

for index, row in df_sorted.iterrows():
    aspect_name_latex = str(row['Aspect']).replace('_', '\\_')
    counts = [str(row[scenario_key]) for scenario_key in file_data.keys()]
    latex_table += f"        {aspect_name_latex} & " + " & ".join(counts) + " \\\\\n"

latex_table += """        \\bottomrule
        \\end{tabular}
    \\vspace{-5pt}
\\end{table}
"""

print(latex_table)

# Save the combined DataFrame to a single CSV
output_csv_path = 'combined_code_reasoning_counts.csv'
df_sorted.to_csv(output_csv_path, index=False)

print(f"\nCombined CSV file '{output_csv_path}' created successfully.")