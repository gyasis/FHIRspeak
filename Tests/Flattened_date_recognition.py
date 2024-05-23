import os
import re
import sys
import csv

def parse_dates_with_warnings(text):
    # Define regex patterns for all date types
    effective_pattern = r"effective Date Time is (\d{4}-\d{2}-\d{2})T"
    issued_pattern = r"issued is (\d{4}-\d{2}-\d{2})T"
    onset_pattern = r"onset Date Time is (\d{4}-\d{2}-\d{2})T"
    recorded_pattern = r"recorded Date is (\d{4}-\d{2}-\d{2})T"
    context_start_pattern = r"context period start is (\d{4}-\d{2}-\d{2})T"
    context_end_pattern = r"context period end is (\d{4}-\d{2}-\d{2})T"
    period_start_pattern = r"period start is (\d{4}-\d{2}-\d{2})T"
    participant_end_pattern = r"period end is (\d{4}-\d{2}-\d{2})T"
    
    # Search for patterns and extract the date part
    effective_date = re.search(effective_pattern, text)
    issued_date = re.search(issued_pattern, text)
    onset_date = re.search(onset_pattern, text)
    recorded_date = re.search(recorded_pattern, text)
    context_start_date = re.search(context_start_pattern, text)
    context_end_date = re.search(context_end_pattern, text)
    period_start_date = re.search(period_start_pattern, text)
    participant_end_date = re.search(participant_end_pattern, text)
    
    # Extract the date or None if not found
    effective_str = effective_date.group(1) if effective_date else None
    issued_str = issued_date.group(1) if issued_date else None
    onset_str = onset_date.group(1) if onset_date else None
    recorded_str = recorded_date.group(1) if recorded_date else None
    context_start_str = context_start_date.group(1) if context_start_date else None
    context_end_str = context_end_date.group(1) if context_end_date else None
    period_start_str = period_start_date.group(1) if period_start_date else None
    participant_end_str = participant_end_date.group(1) if participant_end_date else None
    
    return effective_str, issued_str, onset_str, recorded_str, context_start_str, context_end_str, period_start_str, participant_end_str

def process_files(folder_path):
    results = []
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Check if the file is a text file
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                dates = parse_dates_with_warnings(content)
                results.append([filename] + list(dates))
    return results

def output_to_csv(results, output_csv):
    # Write results to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'Effective Date', 'Issued Date', 'Onset Date', 'Recorded Date', 'Context Period Start', 'Context Period End', 'Period Start', 'Participant 0 Period End'])
        csvwriter.writerows(results)

if __name__ == "__main__":
    folder_path = sys.argv[1]
    output_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results.csv')

    results = process_files(folder_path)

    if '--csv' in sys.argv:
        output_to_csv(results, output_csv)
        print(f"Results have been written to {output_csv}")
    else:
        for result in results:
            print(f"Filename: {result[0]}, Effective Date: {result[1]}, Issued Date: {result[2]}, Onset Date: {result[3]}, Recorded Date: {result[4]}, Context Period Start: {result[5]}, Context Period End: {result[6]}, Period Start: {result[7]}, Participant 0 Period End: {result[8]}")
