//! Helper functions: CSV parsing, metrics, etc.

use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Load CSV file into (features, targets)
/// Assumes last column is target.
pub fn load_csv(path: &Path) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let values: Vec<f64> = line
            .split(',')
            .map(|v| v.trim().parse::<f64>().unwrap())
            .collect();

        let n = values.len();
        if n < 2 {
            return Err("Each row must have at least one feature and one target".into());
        }

        let (features, target) = values.split_at(n - 1);
        x_data.push(features.to_vec());
        y_data.push(target[0]);
    }

    Ok((x_data, y_data))
}

/// Compute Mean Squared Error
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len();
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n as f64
}

/// Display a simple accuracy matrix
pub fn display_accuracy_matrix(y_true: &[f64], y_pred: &[f64]) {
    let total = y_true.len();
    
    // Count predictions within different thresholds
    let within_5_percent = y_true.iter().zip(y_pred.iter())
        .filter(|(true_val, pred_val)| {
            let error_percent = if **true_val != 0.0 {
                ((**true_val - **pred_val).abs() / (**true_val).abs()) * 100.0
            } else {
                (**true_val - **pred_val).abs()
            };
            error_percent <= 5.0
        }).count();

    let within_10_percent = y_true.iter().zip(y_pred.iter())
        .filter(|(true_val, pred_val)| {
            let error_percent = if **true_val != 0.0 {
                ((**true_val - **pred_val).abs() / (**true_val).abs()) * 100.0
            } else {
                (**true_val - **pred_val).abs()
            };
            error_percent <= 10.0
        }).count();
    println!("\n=== Accuracy Matrix ===");
    println!("Total Predictions: {}", total);
    println!("┌─────────────────┬─────────┬─────────┐");
    println!("│ Threshold       │ Correct │ Accuracy│");
    println!("├─────────────────┼─────────┼─────────┤");
    println!("│ ±5%             │ {:>7} │ {:>6.1}% │", within_5_percent, (within_5_percent as f64 / total as f64) * 100.0);
    println!("│ ±10%            │ {:>7} │ {:>6.1}% │", within_10_percent, (within_10_percent as f64 / total as f64) * 100.0);
    println!("└─────────────────┴─────────┴─────────┘");
}