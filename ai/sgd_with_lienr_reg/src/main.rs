//! A minimal CLI for Linear Regression using Stochastic Gradient Descent.
//!
//! Commands:
//!   train <csv_path> <learning_rate> <epochs> <output_model>
//!   test <csv_path> <model_path>
//!   predict <model_path> <feature1> <feature2> ...
//!
//! The CSV format expects: each row = feature1,feature2,...,target
//! The last column is the target value.

mod model;
mod utils;

use clap::{Parser, Subcommand};
use model::LinearRegression;
use utils::{load_csv, mean_squared_error};
use std::path::PathBuf;

use crate::utils::display_accuracy_matrix;

/// Simple Linear Regression with SGD CLI.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model on a CSV file
    Train {
        /// Input CSV path
        csv: PathBuf,
        /// Learning rate
        #[arg(short, long, default_value_t = 0.001)]
        lr: f64,
        /// Number of epochs
        #[arg(short, long, default_value_t = 1000)]
        epochs: usize,
        /// Output model file path
        #[arg(short, long, default_value = "model.json")]
        out: PathBuf,
    },

    /// Test a model and compute MSE on dataset
    Test {
        /// Input CSV path
        csv: PathBuf,
        /// Model file path
        #[arg(short, long, default_value = "model.json")]
        model: PathBuf,
    },

    /// Predict an output from raw features using a trained model
    Predict {
        /// Model file path
        #[arg(short, long, default_value = "model.json")]
        model: PathBuf,
        /// Feature values
        features: Vec<f64>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train { csv, lr, epochs, out } => {
            println!("Loading data from {}", csv.display());
            let (x, y) = load_csv(&csv).expect("Failed to load CSV");
            println!("Training model (lr={}, epochs={})...", lr, epochs);
            let mut model = LinearRegression::new(x[0].len(), lr);
            println!("{:?}", model);
            model.train(&x, &y, epochs);

            model.save(&out).expect("Failed to save model");
            println!("Model saved to {}", out.display());
        }

        Commands::Test { csv, model } => {
            let (x, y) = load_csv(&csv).expect("Failed to load CSV");
            let model = LinearRegression::load(&model).expect("Failed to load model");
            let preds: Vec<f64> = x.iter().map(|row| model.predict(row)).collect();
            let mse = mean_squared_error(&y, &preds);
            
            println!("Mean Squared Error: {:.6}", mse);
            display_accuracy_matrix(&y, &preds);
        }

        Commands::Predict { model, features } => {
            let model = LinearRegression::load(&model).expect("Failed to load model");
            let pred = model.predict(&features);
            println!("Prediction: {:.4}", pred);
        }
    }
}
