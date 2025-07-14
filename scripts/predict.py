import argparse
import pandas as pd
import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.prediction_pipeline import PredictionPipeline

def main(args):
    """
    Main function to load the pipeline, run predictions, and save the results.
    """
    if not os.path.exists(args.pipeline_path):
        print(f"Error: Pipeline file not found at '{args.pipeline_path}'")
        sys.exit(1)

    if not os.path.exists(args.input_path):
        print(f"Error: Input data file not found at '{args.input_path}'")
        sys.exit(1)

    # Load the trained pipeline
    print(f"Loading prediction pipeline from {args.pipeline_path}...")
    pipeline = PredictionPipeline.load(args.pipeline_path)
    print("Pipeline loaded successfully.")

    # Load the raw data for prediction
    print(f"Loading input data from {args.input_path}...")
    try:
        # Assuming the input is a CSV file, adjust if other formats are expected
        X_raw = pd.read_csv(args.input_path)
    except Exception as e:
        print(f"Error loading input data: {e}")
        sys.exit(1)

    print(f"Input data loaded successfully with shape {X_raw.shape}.")

    # Make predictions
    print("Running predictions...")
    try:
        dt_estimation_df, ls_estimation_df = pipeline.predict(X_raw)
        print("Predictions generated successfully.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save the results
    os.makedirs(args.output_dir, exist_ok=True)

    dt_output_path = os.path.join(args.output_dir, "DTestimation.csv")
    ls_output_path = os.path.join(args.output_dir, "LSestimation.csv")

    try:
        dt_estimation_df.to_csv(dt_output_path, index=False)
        print(f"DTestimation results saved to {dt_output_path}")

        ls_estimation_df.to_csv(ls_output_path, index=False)
        print(f"LSestimation results saved to {ls_output_path}")
    except Exception as e:
        print(f"Error saving prediction results: {e}")
        sys.exit(1)

    print("\nPrediction script finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run prediction using a trained pipeline.")

    parser.add_argument('--pipeline_path', type=str, required=True,
                        help='Path to the saved prediction pipeline file (e.g., prediction_pipeline.joblib).')

    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input CSV file with data for prediction.')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the output prediction files (DTestimation.csv, LSestimation.csv) will be saved.')

    args = parser.parse_args()
    main(args)
