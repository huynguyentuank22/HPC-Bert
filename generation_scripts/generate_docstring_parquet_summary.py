import pandas as pd 
import os

if __name__ == "__main__":
    # Load dataset 
    df = pd.read_csv("docs/parquet_files_summary.csv")
    columns = df.columns.tolist()

    md_file = "docs/parquet_file_summary.md"
        
    header = "|"+ "|".join(columns) + "|\n"
    header += "|" + "|".join(["-"*len(col) for col in columns]) + "|\n"

    docstring = "\n".join(
        "|" + "|".join(str(row[col]) for col in columns) + "|"
        for _, row in df.iterrows()
    )
    
    with open(md_file, "w") as f:
        f.write("# Parquet Files Summary\n\n")
        f.write(header)
        for l in docstring:
            f.write(l)
                    
        