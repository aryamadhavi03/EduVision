import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to analyze the dataset
def analyze_dataset(file_path):
    # Load the CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset from {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please check the file path.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Create a directory for output files if it doesn't exist
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    # Open a file to save the analysis
    with open(os.path.join(output_dir, "dataset_analysis.txt"), "w") as f:

        # 1. Basic Dataset Information
        f.write("=== Basic Dataset Information ===\n")
        f.write(f"Dataset Shape (Rows, Columns): {df.shape}\n")
        f.write("\nDataset Info:\n")
        df.info(buf=f)
        f.write("\nFirst 5 Rows:\n")
        f.write(df.head().to_string() + "\n\n")

        # 2. Summary Statistics
        f.write("=== Summary Statistics ===\n")
        f.write(df.describe().to_string() + "\n\n")

        # 3. Missing Values
        f.write("=== Missing Values ===\n")
        missing_values = df.isnull().sum()
        f.write(missing_values.to_string() + "\n")
        f.write(f"Total rows with any missing value: {df.isnull().any(axis=1).sum()}\n\n")

        # 4. Duplicate Rows
        f.write("=== Duplicate Rows ===\n")
        duplicate_count = df.duplicated().sum()
        f.write(f"Number of duplicate rows: {duplicate_count}\n")
        if duplicate_count > 0:
            f.write("Duplicate rows:\n")
            f.write(df[df.duplicated()].to_string() + "\n\n")
        else:
            f.write("No duplicate rows found.\n\n")

        # 5. Distribution of Recommendation
        f.write("=== Distribution of Recommendation ===\n")
        recommendation_counts = df["Recommendation"].value_counts()
        f.write(recommendation_counts.to_string() + "\n")
        # Calculate and write percentage distribution for each category
        percentage_dist = 100 * recommendation_counts / len(df)
        f.write("Percentage Distribution:\n")
        for category, percent in percentage_dist.items():
            f.write(f"{category}: {percent:.2f}%\n")
        f.write("\n")

        # 6. Distribution of Numeric Features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            f.write("=== Distribution of Numeric Features ===\n")
            df[numeric_cols].hist(bins=20, figsize=(12, 8))
            plt.suptitle('Distribution of Numeric Features')
            plt.savefig(os.path.join(output_dir, "feature_distributions.png"))
            plt.close()
            f.write("Histograms saved as 'feature_distributions.png'\n\n")
        else:
            f.write("No numeric features found for distribution analysis.\n\n")

        # 7. Box Plots for Outliers
        if len(numeric_cols) > 0:
            f.write("=== Box Plots for Outlier Detection ===\n")
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df[numeric_cols])
            plt.title('Box Plot of Numeric Features')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(output_dir, "box_plots.png"))
            plt.close()
            f.write("Box plots saved as 'box_plots.png'\n\n")
        else:
            f.write("No numeric features found for box plot analysis.\n\n")

        # 8. Correlation Matrix
        if len(numeric_cols) > 1:
            f.write("=== Correlation Matrix ===\n")
            correlation_matrix = df[numeric_cols].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix of Numeric Features')
            plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
            plt.close()
            f.write("Correlation matrix saved as 'correlation_matrix.png'\n\n")
        else:
            f.write("Insufficient numeric features for correlation analysis.\n\n")

        # 9. Summary
        f.write("=== Analysis Summary ===\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Total columns: {len(df.columns)}\n")
        f.write(f"Balanced categories: {max(recommendation_counts) - min(recommendation_counts) <= 1}\n")
        f.write("Analysis completed. Check 'analysis_output' directory for files.\n")

    print(f"Analysis saved to {os.path.join(output_dir, 'dataset_analysis.txt')} and associated plots.")

# Run the analysis
if __name__ == "__main__":
    file_path = "balanced_student_data.csv"  # Default file path; change as needed
    analyze_dataset(file_path)