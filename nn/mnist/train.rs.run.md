(.venv) cccuser@cccimacdeiMac ai4 % cargo run --bin mnist_train
warning: hiding a lifetime that's elided elsewhere is confusing
  --> nn/tensor.rs:92:17
   |
92 | ...ta(&self) -> std::cell::Ref<Vec<f64>> {
   |       ^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^ the same lifetime is hidden here
   |       |
   |       the lifetime is elided here
   |
   = help: the same lifetime is referred to in inconsistent ways, making the signature confusing
   = note: `#[warn(mismatched_lifetime_syntaxes)]` on by default
help: use `'_` for type paths
   |
92 |     pub fn data(&self) -> std::cell::Ref<'_, Vec<f64>> {
   |                                          +++

warning: hiding a lifetime that's elided elsewhere is confusing
  --> nn/tensor.rs:95:21
   |
95 | ...ut(&self) -> std::cell::RefMut<Vec<f64>> {
   |       ^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ the same lifetime is hidden here
   |       |
   |       the lifetime is elided here
   |
   = help: the same lifetime is referred to in inconsistent ways, making the signature confusing
help: use `'_` for type paths
   |
95 |     pub fn data_mut(&self) -> std::cell::RefMut<'_, Vec<f64>> {
   |                                                 +++

warning: hiding a lifetime that's elided elsewhere is confusing
  --> nn/tensor.rs:98:17
   |
98 | ...ad(&self) -> std::cell::Ref<Vec<f64>> {
   |       ^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^ the same lifetime is hidden here
   |       |
   |       the lifetime is elided here
   |
   = help: the same lifetime is referred to in inconsistent ways, making the signature confusing
help: use `'_` for type paths
   |
98 |     pub fn grad(&self) -> std::cell::Ref<'_, Vec<f64>> {
   |                                          +++

warning: hiding a lifetime that's elided elsewhere is confusing
   --> nn/tensor.rs:101:21
    |
101 | ...ut(&self) -> std::cell::RefMut<Vec<f64>> {
    |       ^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ the same lifetime is hidden here
    |       |
    |       the lifetime is elided here
    |
    = help: the same lifetime is referred to in inconsistent ways, making the signature confusing
help: use `'_` for type paths
    |
101 |     pub fn grad_mut(&self) -> std::cell::RefMut<'_, Vec<f64>> {
    |                                                 +++

warning: `ai4` (lib) generated 4 warnings (run `cargo fix --lib -p ai4` to apply 4 suggestions)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.17s
     Running `target/debug/mnist_train`
Loading MNIST dataset...
Dataset: 60000 samples
Epoch 1 Batch 0 Loss: 2.5909
Epoch 1 Batch 1 Loss: 2.1983
Epoch 1 Batch 2 Loss: 2.2417
Epoch 1 Batch 3 Loss: 1.9308
Epoch 1 Batch 4 Loss: 1.7571
Epoch 1 Batch 5 Loss: 1.6565
Epoch 1 Batch 6 Loss: 1.5163
Epoch 1 Batch 7 Loss: 1.5039
Epoch 1 Batch 8 Loss: 1.3223
Epoch 1 Batch 9 Loss: 1.3454
Epoch 1 Batch 10 Loss: 1.2083
Epoch 1 Batch 11 Loss: 1.0462
Epoch 1 Batch 12 Loss: 0.9108
Epoch 1 Batch 13 Loss: 0.8986
Epoch 1 Batch 14 Loss: 0.9410
Epoch 1 Batch 15 Loss: 0.7481
Epoch 1 Accuracy: 55.18%
Epoch 2 Batch 0 Loss: 0.6783
Epoch 2 Batch 1 Loss: 0.4545
Epoch 2 Batch 2 Loss: 0.7304
Epoch 2 Batch 3 Loss: 0.4046
Epoch 2 Batch 4 Loss: 0.3390
Epoch 2 Batch 5 Loss: 0.3956
Epoch 2 Batch 6 Loss: 0.4778
Epoch 2 Batch 7 Loss: 0.6850
Epoch 2 Batch 8 Loss: 0.3666
Epoch 2 Batch 9 Loss: 0.5726
Epoch 2 Batch 10 Loss: 0.3953
Epoch 2 Batch 11 Loss: 0.4363
Epoch 2 Batch 12 Loss: 0.4404
Epoch 2 Batch 13 Loss: 0.4881
Epoch 2 Batch 14 Loss: 0.4292
Epoch 2 Batch 15 Loss: 0.4190
Epoch 2 Accuracy: 85.84%
Epoch 3 Batch 0 Loss: 0.3405
Epoch 3 Batch 1 Loss: 0.2566
Epoch 3 Batch 2 Loss: 0.4600
Epoch 3 Batch 3 Loss: 0.2504
Epoch 3 Batch 4 Loss: 0.2177
Epoch 3 Batch 5 Loss: 0.2104
Epoch 3 Batch 6 Loss: 0.2508
Epoch 3 Batch 7 Loss: 0.4357
Epoch 3 Batch 8 Loss: 0.1824
Epoch 3 Batch 9 Loss: 0.2687
Epoch 3 Batch 10 Loss: 0.1775
Epoch 3 Batch 11 Loss: 0.2021
Epoch 3 Batch 12 Loss: 0.2936
Epoch 3 Batch 13 Loss: 0.3746
Epoch 3 Batch 14 Loss: 0.2595
Epoch 3 Batch 15 Loss: 0.2700
Epoch 3 Accuracy: 92.29%
Model saved to nn/mnist/model.json
Training complete!