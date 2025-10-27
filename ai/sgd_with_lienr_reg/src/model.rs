use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Read, Write};
use rand::Rng;
use rand::seq::SliceRandom; 
use rand::thread_rng;

#[derive(Debug,Serialize, Deserialize)]
pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub lr: f64,
}

impl LinearRegression {
    
    pub fn new(n_features: usize, lr: f64) -> Self {
        let mut rng = rand::thread_rng();

        Self {
            weights: (0..n_features).map(|_| rng.gen_range(-0.1..0.1)).collect(),
            bias: rng.gen_range(-0.1..0.1),
            lr,
        }
    }
    
    pub fn predict(&self, features: &[f64]) -> f64 {

        self.weights
            .iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias

    }

    pub fn train(&mut self, x: &[Vec<f64>], y: &[f64], epochs: usize) {
        let n = x.len();
        let mut rng = thread_rng();
        
        let mut indices: Vec<usize> = (0..n).collect();

        for epoch in 0..epochs {

            indices.shuffle(&mut rng);

            let mut total_loss = 0.0;
            
            for &i in &indices {
                let y_pred = self.predict(&x[i]);
                let error = y[i] - y_pred;
                total_loss += error * error;

                for j in 0..self.weights.len() {
                    self.weights[j] += self.lr * 2.0 * error * x[i][j];
                }
                self.bias += self.lr * 2.0 * error;
                
                println!("{:.5?}", self.weights);
                println!("{:.5}",self.bias);
            }

            if (epoch + 1) % 100 == 0 {
                let mse = total_loss / n as f64;
                println!("Epoch {:>5}: MSE = {}", epoch + 1, mse);
            }
        }
    }

    // --- MODIFIED SAVE/LOAD TO AVOID .unwrap() ---

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        // Use '?' to propagate the error instead of panicking
        let json = serde_json::to_string_pretty(self)?; 
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        // Use '?' to propagate the error instead of panicking
        let model: LinearRegression = serde_json::from_str(&contents)?;
        Ok(model)
    }
}

#[cfg(test)]
mod test {
    use std::f64::NAN;

    use super::*;
    
    #[test]
    fn test_model_predict(){
        let model = LinearRegression::new(1, 0.001);
        let y = model.predict(&vec![0.1]);
        println!("{}",y);
        assert!(!y.is_nan())
    }
}