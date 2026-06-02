import pandas as pd
import os

# Define file paths dynamically relative to the script location
base_dir = os.path.dirname(os.path.abspath(__file__))
target_files = {
    "India": os.path.join(base_dir, "final_results", "aggregated_all_models_india_llm_parsed_fixed.csv"),
    "Turkey": os.path.join(base_dir, "final_results", "aggregated_all_models_turkey_llm_parsed_fixed.csv"),
    "Vietnam": os.path.join(base_dir, "final_results", "aggregated_all_models_vietnam_llm_parsed_fixed.csv")
}

output_report = os.path.join(base_dir, "unparseable_cases_report.txt")

def generate_report():
    report_lines = []
    report_lines.append("UNPARSEABLE CASES REPORT\n" + "="*25 + "\n")
    
    total_unparseable = 0
    
    # Store detailed data for summary table
    summary_data = []

    for country, file_path in target_files.items():
        if not os.path.exists(file_path):
            report_lines.append(f"File not found for {country}: {file_path}\n")
            continue
            
        report_lines.append(f"Country: {country}")
        report_lines.append("-" * (9 + len(country)))
        
        df = pd.read_csv(file_path)
        unparseable_df = df[df['parsed_label'] == 'unparseable']
        
        count = len(unparseable_df)
        total_unparseable += count
        
        if count == 0:
            report_lines.append("No unparseable cases found.\n")
        else:
            # Group by model
            model_groups = unparseable_df.groupby('model_label')
            
            for model_label, group in model_groups:
                unique_scenarios = sorted(group['scenario_id'].unique())
                num_unparseable = len(group)
                report_lines.append(f"  Model: {model_label}")
                report_lines.append(f"    Total unparseable instances: {num_unparseable}")
                report_lines.append(f"    Unique scenario IDs affected ({len(unique_scenarios)}): {unique_scenarios}")
                report_lines.append("")
                
                summary_data.append({
                    "Country": country,
                    "Model": model_label,
                    "Total Instances": num_unparseable,
                    "Unique Scenarios": len(unique_scenarios)
                })
        
        report_lines.append("\n")

    report_lines.append(f"TOTAL UNPARSEABLE INSTANCES ACROSS ALL FILES: {total_unparseable}\n")
    
    # Add a summary table at the end
    if summary_data:
        report_lines.append("SUMMARY TABLE")
        report_lines.append("="*13)
        summary_df = pd.DataFrame(summary_data)
        report_lines.append(summary_df.to_string(index=False))
    
    # Write to file
    with open(output_report, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"Report generated at: {output_report}")
    print(f"Total unparseable instances found: {total_unparseable}")

if __name__ == "__main__":
    generate_report()
