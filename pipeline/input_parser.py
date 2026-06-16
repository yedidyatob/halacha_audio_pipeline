import re
from typing import List

def parse_simanim_string(input_str: str) -> List[int]:
    """
    Parses a string representing numbers and ranges (e.g. "94,95-97, 100") 
    and returns a sorted list of unique positive integers.
    
    Raises ValueError if the input format is invalid, contains negative numbers, 
    or represents non-integer structures.
    """
    if not input_str or not input_str.strip():
        raise ValueError("Input string is empty.")
    
    # Remove all spaces
    clean_str = input_str.replace(" ", "")
    
    # Validate overall allowed characters (digits, commas, hyphens)
    if not re.match(r"^[0-9,-]+$", clean_str):
        raise ValueError("Input contains invalid characters. Only digits, commas, and hyphens are allowed.")
        
    parts = clean_str.split(",")
    simanim = set()
    
    for part in parts:
        if not part:
            raise ValueError("Empty section found (double commas or trailing comma).")
            
        if "-" in part:
            range_parts = part.split("-")
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range format: '{part}'. Ranges must contain exactly one hyphen.")
            
            start_str, end_str = range_parts[0], range_parts[1]
            if not start_str.isdigit() or not end_str.isdigit():
                raise ValueError(f"Invalid range bounds: '{part}'. Bounds must be positive integers.")
                
            start, end = int(start_str), int(end_str)
            if start <= 0 or end <= 0:
                raise ValueError("Siman numbers must be positive integers greater than zero.")
            if start > end:
                raise ValueError(f"Invalid range order: '{part}'. Start of range must be less than or equal to the end.")
                
            for num in range(start, end + 1):
                simanim.add(num)
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid number format: '{part}'. Must be a positive integer.")
            val = int(part)
            if val <= 0:
                raise ValueError("Siman numbers must be positive integers greater than zero.")
            simanim.add(val)
            
    return sorted(list(simanim))
