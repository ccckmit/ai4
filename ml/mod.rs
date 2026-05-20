//! ml - Machine Learning toolkit (sklearn-style).

pub mod linear_models;
pub mod tree;
pub mod ensemble;
pub mod clustering;
pub mod decomposition;
pub mod metrics;
pub mod preprocessing;

pub use linear_models::{LinearRegression, LogisticRegression};
pub use tree::DecisionTree;
pub use ensemble::RandomForest;
pub use clustering::KMeans;
pub use decomposition::PCA;
pub use preprocessing::{StandardScaler, train_test_split};
pub use metrics::{accuracy_score, mean_squared_error, r2_score};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_regression() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let mut model = LinearRegression::new(0.01, 1000);
        model.fit(&x, &y);
        
        let pred = model.predict(&vec![vec![6.0]]);
        assert!((pred[0] - 12.0).abs() < 1.0);
    }
    
    #[test]
    fn test_accuracy() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 2, 0, 0, 2];
        assert_eq!(accuracy_score(&y_true, &y_pred), 5.0 / 6.0);
    }
    
    #[test]
    fn test_logistic_regression() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0], vec![6.0]];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        
        let mut model = LogisticRegression::new(0.1, 1000);
        model.fit(&x, &y);
        
        let pred = model.predict(&x);
        let acc = accuracy_score(&y.iter().map(|&x| x as i32).collect::<Vec<_>>(), &pred);
        assert!(acc > 0.7);
    }
    
    #[test]
    fn test_kmeans() {
        let mut X1 = Vec::new();
        let mut X2 = Vec::new();
        
        for _ in 0..30 {
            X1.push(vec![rand::random::<f64>() * 2.0 + 2.0, rand::random::<f64>() * 2.0 + 2.0]);
            X2.push(vec![rand::random::<f64>() * 2.0 - 2.0, rand::random::<f64>() * 2.0 - 2.0]);
        }
        
        let mut X = X1;
        X.extend(X2);
        
        let mut kmeans = KMeans::new(2, 300, 5);
        kmeans.fit(&X);
        
        let labels = kmeans.predict(&X);
        assert_eq!(labels.len(), X.len());
    }
    
    #[test]
    fn test_pca() {
        let mut X = Vec::new();
        for _ in 0..50 {
            X.push(vec![rand::random(), rand::random(), rand::random(), rand::random()]);
        }
        
        let mut pca = PCA::new(Some(2));
        let X_transformed = pca.fit_transform(&X);
        
        assert_eq!(X_transformed.len(), 50);
        assert_eq!(X_transformed[0].len(), 2);
    }
    
    #[test]
    fn test_scaler() {
        let X = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        
        let mut scaler = StandardScaler::new();
        let X_scaled = scaler.fit_transform(&X);
        
        for col in 0..2 {
            let mean: f64 = X_scaled.iter().map(|row| row[col]).sum::<f64>() / X_scaled.len() as f64;
            assert!((mean - 0.0).abs() < 1e-10);
        }
    }
}