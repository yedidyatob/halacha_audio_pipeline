import re
import sys
from pathlib import Path


def strip_nikkud(input_filepath: str, output_filepath: str = None) -> None:
    """Reads a UTF-8 text file, removes all Hebrew nikkud (vowels)

    and cantillation marks, and writes the output to a file.
    """
    input_path = Path(input_filepath)

    if not input_path.exists():
        print(f"Error: The file '{input_filepath}' does not exist.")
        sys.exit(1)

    # If no output path is provided, append '_no_nikkud' to the original name
    if output_filepath is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_no_nikkud{input_path.suffix}"
        )
    else:
        output_path = Path(output_filepath)

    # Unicode range for Hebrew diacritics (vowels, dagesh, cantillation marks)
    nikkud_pattern = re.compile(r"[\u0591-\u05C7]")

    try:
        # Read with UTF-8 encoding to handle Hebrew correctly
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Remove the nikkud
        clean_text = nikkud_pattern.sub("", text)

        # Write to the new file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        print(f"Success! Cleaned file saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Strip Hebrew nikkud (vowels) and cantillation marks from a text file."
    )
    parser.add_argument(
        "input_file", 
        help="Path to the input Hebrew text file with nikkud"
    )
    parser.add_argument(
        "-o", "--output", 
        dest="output_file",
        help="Path to save the clean text file (optional; defaults to appending '_no_nikkud')"
    )
    args = parser.parse_args()
    strip_nikkud(args.input_file, args.output_file)
