# aspect_keywords = {
#     'Poor Variable/Method Names': ['poor variable', 'poor method names', 'bad variable', 'bad method names', 'inconsistent naming', 'non-standard method name', 'magic numbers', 'magic strings'],
#     'Poor Readability': ['poor readability', 'hard to read', 'difficult to understand', 'unclear', 'less readable', 'hinder readability', 'cryptic', 'verbosity', 'verbose'],
#     'Good Code Structure': ['well-structured', 'good structure', 'well-organized', 'good separation of concerns', 'modular design', 'clean structure', 'logical organization', 'well-organized inner classes', 'well-defined inner classes'],
#     'Good Documentation': ['comprehensive documentation', 'good documentation', 'well-documented', 'JavaDoc comments', 'clear comments', 'descriptive JavaDoc comments', 'detailed comments'],
#     'High Code Complexity': ['complex', 'complexity', 'nested', 'over-engineering', 'overly complex', 'convoluted', 'dense logic'],
#     'Bad Code Structure': ['lacks ... structure', 'bad separation of concerns', 'over-engineering', 'spaghetti code', 'monolithic', 'tight coupling', 'hardcoded data'],
#     'Lack of Documentation/Comments': ['lacks comprehensive comments', 'lacks documentation', 'minimal comments', 'unclear purpose', 'no documentation', 'missing javadoc', 'undocumented'],
#     'Adherence to Best Practices': ['best practices', 'Java conventions', 'object-oriented principles', 'good practices', 'Java idioms', 'follows standard Java conventions', 'strong adherence to best practices'],
#     'Good Implementation/Logic': ['excellent implementation', 'good implementation', 'clear method implementations', 'thoughtful design', 'clean implementation', 'straightforward', 'logical', 'efficient'],
#     'Consistent Formatting': ['consistent formatting', 'well-formatted', 'consistent styling', 'consistent indentation'],
#     'Good Variable/Method Names': ['clear method names', 'meaningful variable names', 'descriptive ... names', 'consistent naming'],
#     'Good Error Handling': ['robust error handling', 'proper exception handling', 'handles potential exceptions'],
#     'Poor Error Handling': ['error handling is minimal', 'swallowed exception', 'suppressed empty catch block', 'inconsistent ... error handling', 'ignores exceptions', 'empty catch block'],
#     'Good Readability': ['high readability', 'good readability', 'clear', 'easy to understand', 'readable', 'self-explanatory', 'easy to follow'],
#     'Unprofessional Comments': ['unprofessional'],
#     'Inconsistent Formatting': ['inconsistent line breaks', 'inconsistent formatting'],
#     'Good Use of Language Features': ['modern Java features', 'switch expressions', 'lambda', 'streams', 'Java generics', 'temporal interfaces']
# }

import re
import pandas as pd

# The file path and aspect_keywords dictionary remain the same
file_path = 'cleancode_reasoning.txt'

aspect_keywords = {
    "Poor Variable/Method Names": [
        "poor variable",
        "poor method names",
        "bad variable",
        "bad method names",
        "inconsistent naming",
        "non-standard method name",
        "magic numbers",
        "magic strings",
        "unconventional naming",
        "confusing names",
        "cryptic names",
        "single-letter names",
        "food-related names",
        "vegetable-themed variable names",
        "spice-related names",
        "culinary terms",
        "nonsensical names",
        "arbitrary naming"
    ],
    "Poor Readability": [
        "poor readability",
        "hard to read",
        "difficult to understand",
        "unclear",
        "less readable",
        "hinder readability",
        "cryptic",
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
        "JavaDoc comments",
        "clear comments",
        "descriptive JavaDoc comments",
        "detailed comments",
        "comprehensive Javadoc",
        "clear method comments",
        "detailed JavaDoc comments"
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
    "Lack of Documentation/Comments": [
        "lacks comprehensive comments",
        "lacks documentation",
        "minimal comments",
        "unclear purpose",
        "no documentation",
        "missing javadoc",
        "undocumented",
        "lacks meaningful comments",
        "lack of clear comments"
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

# NEW: Define the pairs of negative and positive aspects
negative_to_positive_pairs = {
    "Poor Variable/Method Names": "Good Variable/Method Names",
    "Poor Readability": "Good Readability",
    "Bad Code Structure": "Good Code Structure",
    "Lack of Documentation/Comments": "Good Documentation",
    "Poor Error Handling": "Good Error Handling",
    "Inconsistent Formatting": "Consistent Formatting",
}

aspect_counts = {aspect: 0 for aspect in aspect_keywords.keys()}

with open(file_path, 'r') as f:
    for line in f:
        # Pre-process the line to make it lowercase and remove source tags
        line_content = re.sub(r'\\', '', line).strip().lower()

        if not line_content:
            continue

        # NEW: First, find all aspects that match the current line
        matched_aspects_in_line = set()
        for aspect, keywords in aspect_keywords.items():
            for keyword in keywords:
                # Handle special patterns like "..."
                if '...' in keyword:
                    # Creates a regex to match any word between the start and end of the keyword
                    pattern = r'\b' + re.escape(keyword).replace(r'\.\.\.', r'\s*\w*\s*') + r'\b'
                else:
                    pattern = r'\b' + re.escape(keyword) + r'\b'

                if re.search(pattern, line_content):
                    matched_aspects_in_line.add(aspect)
                    break  # Optimization: once an aspect is found, no need to check its other keywords

        # NEW: Logic to exclude positive counts if a negative counterpart is present (to prevent false positive)
        final_aspects_to_count = set(matched_aspects_in_line)

        for negative_aspect, positive_aspect in negative_to_positive_pairs.items():
            # If both the negative and its corresponding positive aspect are in the line
            if negative_aspect in matched_aspects_in_line and positive_aspect in matched_aspects_in_line:
                # Remove the positive one from the set of aspects to be counted for this line
                final_aspects_to_count.discard(positive_aspect)

        # Increment the counts for the final filtered set
        for aspect in final_aspects_to_count:
            aspect_counts[aspect] += 1

# Create the LaTeX table (this part remains unchanged)
latex_table = """
\\begin{table}[hbt]
    \\centering
    \\tiny
    \\caption{Code Evaluation}
        \\begin{tabular}{lc}
        \\toprule
        \\textbf{Aspect} & \\textbf{Scenario CC Count} \\\\
        \\midrule
"""

# Sort the aspects for a consistent table order
sorted_aspects = sorted(aspect_counts.keys())

for aspect in sorted_aspects:
    count = aspect_counts[aspect]
    # Escape underscores for LaTeX
    aspect_name_latex = aspect.replace('_', '\\_')
    latex_table += f"        {aspect_name_latex} & {count} \\\\\n"

latex_table += """        \\bottomrule
        \\end{tabular}
    \\vspace{-5pt}
\\end{table}
"""

print(latex_table)

# Create and save the DataFrame (this part remains unchanged)
df = pd.DataFrame(list(aspect_counts.items()), columns=['Aspect', 'Scenario CC Count'])
df_sorted = df.sort_values(by='Aspect').reset_index(drop=True)
df_sorted.to_csv('code_evaluation_counts.csv', index=False)

print("\\nCSV file 'code_evaluation_counts.csv' created successfully.")