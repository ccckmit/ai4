use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use rand::SeedableRng;

// 使用 AtomicUsize 替代 unsafe 的 static mut，確保執行緒安全並產生唯一 ID
static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// 內部的資料結構，儲存真正的數值、梯度與計算圖關聯
pub struct TensorInner {
    pub id: usize,
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub _backward: Option<Box<dyn Fn()>>,
    pub _prev: Vec<Tensor>, // 直接儲存父節點的引用，不再依賴全域 Registry
}

/// 外部的包裝，這是一個輕量級指標 (Handle)，Clone 只會複製指標而不會複製資料
#[derive(Clone)]
pub struct Tensor {
    inner: Rc<RefCell<TensorInner>>,
}

impl Tensor {
    /// 基礎建構函數
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let len = data.len();
        let expected_len: usize = shape.iter().product();
        assert_eq!(len, expected_len, "Data length must match shape product");

        Self {
            inner: Rc::new(RefCell::new(TensorInner {
                id: TENSOR_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
                data,
                grad: vec![0.0; len],
                shape,
                requires_grad,
                _backward: None,
                _prev: Vec::new(),
            })),
        }
    }

    pub fn id(&self) -> usize {
        self.inner.borrow().id
    }

    pub fn from_vec(data: Vec<f32>, requires_grad: bool) -> Self {
        let shape = vec![data.len()];
        Self::new(data, shape, requires_grad)
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let size = shape.iter().product::<usize>();
        Self::new(vec![0.0; size], shape.to_vec(), requires_grad)
    }

    pub fn randn(shape: &[usize], requires_grad: bool) -> Self {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::from_entropy();
        let size = shape.iter().product::<usize>();
        let data: Vec<f32> = (0..size)
            .map(|_| rng.sample(rand_distr::Normal::new(0.0, 1.0).unwrap()) as f32)
            .collect();
        Self::new(data, shape.to_vec(), requires_grad)
    }

    pub fn zero_grad(&self) {
        let mut inner = self.inner.borrow_mut();
        for v in inner.grad.iter_mut() {
            *v = 0.0;
        }
    }

    /// 反向傳播引擎 (Autograd 核心)
    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        // 深度優先搜尋 (DFS) 建立拓撲排序
        fn build_topo(t: &Tensor, visited: &mut HashSet<usize>, topo: &mut Vec<Tensor>) {
            let id = t.id();
            if !visited.contains(&id) {
                visited.insert(id);
                for prev in t.inner.borrow()._prev.iter() {
                    build_topo(prev, visited, topo);
                }
                topo.push(t.clone());
            }
        }

        build_topo(self, &mut visited, &mut topo);

        // 將起始節點 (通常是 Loss) 的梯度設為 1.0
        {
            let mut inner = self.inner.borrow_mut();
            for v in inner.grad.iter_mut() {
                *v = 1.0;
            }
        }

        // 沿著拓撲排序反向傳遞梯度
        for t in topo.into_iter().rev() {
            // 使用 take() 拿出 _backward 閉包，避免借用衝突 (Borrow Checker Panic)
            let backward_fn = t.inner.borrow_mut()._backward.take();
            if let Some(bw) = backward_fn {
                bw();
            }
        }
    }

    /* ----------------------------------------------------
       數學運算與計算圖建構 (Forward & Backward)
       ---------------------------------------------------- */

    pub fn add(&self, other: &Tensor) -> Tensor {
        let s = self.inner.borrow();
        let o = other.inner.borrow();
        assert_eq!(s.shape, o.shape, "Shapes must match for add");

        let data: Vec<f32> = s.data.iter().zip(o.data.iter()).map(|(a, b)| a + b).collect();
        let requires_grad = s.requires_grad || o.requires_grad;

        let out = Tensor::new(data, s.shape.clone(), requires_grad);
        out.inner.borrow_mut()._prev = vec![self.clone(), other.clone()];

        if requires_grad {
            let self_c = self.clone();
            let other_c = other.clone();
            let out_c = out.clone();

            out.inner.borrow_mut()._backward = Some(Box::new(move || {
                let out_grad = out_c.inner.borrow().grad.clone();
                let mut sg = self_c.inner.borrow_mut();
                let mut og = other_c.inner.borrow_mut();

                for i in 0..out_grad.len() {
                    if sg.requires_grad { sg.grad[i] += out_grad[i]; }
                    if og.requires_grad { og.grad[i] += out_grad[i]; } // [修正] 之前寫成了 og 自己加自己
                }
            }));
        }
        out
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let s = self.inner.borrow();
        let o = other.inner.borrow();
        assert_eq!(s.shape, o.shape, "Shapes must match for mul");

        let data: Vec<f32> = s.data.iter().zip(o.data.iter()).map(|(a, b)| a * b).collect();
        let requires_grad = s.requires_grad || o.requires_grad;

        let out = Tensor::new(data, s.shape.clone(), requires_grad);
        out.inner.borrow_mut()._prev = vec![self.clone(), other.clone()];

        if requires_grad {
            let self_c = self.clone();
            let other_c = other.clone();
            let out_c = out.clone();

            out.inner.borrow_mut()._backward = Some(Box::new(move || {
                let out_g = out_c.inner.borrow().grad.clone();
                let s_d = self_c.inner.borrow().data.clone();
                let o_d = other_c.inner.borrow().data.clone();
                
                let mut sg = self_c.inner.borrow_mut();
                let mut og = other_c.inner.borrow_mut();

                for i in 0..out_g.len() {
                    if sg.requires_grad { sg.grad[i] += out_g[i] * o_d[i]; }
                    if og.requires_grad { og.grad[i] += out_g[i] * s_d[i]; }
                }
            }));
        }
        out
    }

    /// 2D 矩陣乘法 (Matrix Multiplication)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let s = self.inner.borrow();
        let o = other.inner.borrow();
        assert_eq!(s.shape.len(), 2, "Matmul only supports 2D for now");
        assert_eq!(o.shape.len(), 2, "Matmul only supports 2D for now");
        
        let m = s.shape[0];
        let k1 = s.shape[1];
        let k2 = o.shape[0];
        let n = o.shape[1];
        assert_eq!(k1, k2, "Inner dimensions must match");

        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for k in 0..k1 {
                    data[i * n + j] += s.data[i * k1 + k] * o.data[k * n + j];
                }
            }
        }

        let requires_grad = s.requires_grad || o.requires_grad;
        let out = Tensor::new(data, vec![m, n], requires_grad);
        out.inner.borrow_mut()._prev = vec![self.clone(), other.clone()];

        if requires_grad {
            let self_c = self.clone();
            let other_c = other.clone();
            let out_c = out.clone();

            out.inner.borrow_mut()._backward = Some(Box::new(move || {
                let out_g = out_c.inner.borrow().grad.clone();
                let s_d = self_c.inner.borrow().data.clone();
                let o_d = other_c.inner.borrow().data.clone();
                
                let mut sg = self_c.inner.borrow_mut();
                let mut og = other_c.inner.borrow_mut();

                // dA = dC * B^T
                if sg.requires_grad {
                    for i in 0..m {
                        for j in 0..k1 {
                            for l in 0..n {
                                sg.grad[i * k1 + j] += out_g[i * n + l] * o_d[j * n + l];
                            }
                        }
                    }
                }
                // dB = A^T * dC
                if og.requires_grad {
                    for i in 0..k1 {
                        for j in 0..n {
                            for l in 0..m {
                                og.grad[i * n + j] += s_d[l * k1 + i] * out_g[l * n + j];
                            }
                        }
                    }
                }
            }));
        }
        out
    }

    pub fn relu(&self) -> Tensor {
        let s = self.inner.borrow();
        let data: Vec<f32> = s.data.iter().map(|&x| x.max(0.0)).collect();

        let out = Tensor::new(data, s.shape.clone(), s.requires_grad);
        out.inner.borrow_mut()._prev = vec![self.clone()];

        if s.requires_grad {
            let self_c = self.clone();
            let out_c = out.clone();

            out.inner.borrow_mut()._backward = Some(Box::new(move || {
                let out_g = out_c.inner.borrow().grad.clone();
                let s_d = self_c.inner.borrow().data.clone();
                let mut sg = self_c.inner.borrow_mut();

                for i in 0..sg.grad.len() {
                    if s_d[i] > 0.0 {
                        sg.grad[i] += out_g[i];
                    }
                }
            }));
        }
        out
    }

    /// (Batched) 交叉熵損失與內建 Softmax
    /// 要求 self 的 shape 必須為 2D: [batch_size, num_classes]
    pub fn cross_entropy(&self, targets: &[usize]) -> Tensor {
        let s = self.inner.borrow();
        assert_eq!(s.shape.len(), 2, "Cross entropy requires 2D output [batch, classes]");
        let batch_size = s.shape[0];
        let num_classes = s.shape[1];
        assert_eq!(targets.len(), batch_size, "Targets length must match batch size");

        let mut loss = 0.0;
        let mut probs = vec![0.0; s.data.len()];

        for i in 0..batch_size {
            let offset = i * num_classes;
            let logits = &s.data[offset..offset + num_classes];
            
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();

            for j in 0..num_classes {
                probs[offset + j] = exps[j] / sum_exp;
            }

            let target = targets[i];
            loss += -probs[offset + target].max(1e-7).ln();
        }
        
        loss /= batch_size as f32;

        let out = Tensor::new(vec![loss], vec![1], s.requires_grad);
        out.inner.borrow_mut()._prev = vec![self.clone()];

        if s.requires_grad {
            let self_c = self.clone();
            let out_c = out.clone();
            let targets_c = targets.to_vec();

            out.inner.borrow_mut()._backward = Some(Box::new(move || {
                let out_g = out_c.inner.borrow().grad[0];
                let mut sg = self_c.inner.borrow_mut();

                let d_loss = out_g / batch_size as f32;
                
                for i in 0..batch_size {
                    let offset = i * num_classes;
                    let target = targets_c[i];
                    for j in 0..num_classes {
                        let mut p = probs[offset + j];
                        if j == target {
                            p -= 1.0;
                        }
                        sg.grad[offset + j] += p * d_loss;
                    }
                }
            }));
        }
        out
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.borrow();
        write!(f, "Tensor(shape={:?}, data={:?}, requires_grad={})", inner.shape, inner.data, inner.requires_grad)
    }
}
/*
// ============================================
// 以下是一個簡單的 main 函數用來測試功能是否正常
// ============================================
fn main() {
    // 建立權重與輸入 (需要計算梯度的權重設為 true)
    let weights = Tensor::new(vec![0.5, -0.2, 0.1, 0.8, 0.4, -0.5], vec![2, 3], true);
    let inputs = Tensor::new(vec![1.0, 2.0], vec![1, 2], false); // shape: [1, 2]

    // 進行前向傳播 (Forward Pass)
    // outputs: [1, 2] x [2, 3] -> [1, 3]
    let hidden = inputs.matmul(&weights);
    let activated = hidden.relu();
    
    // 計算 Loss
    let targets = vec![2]; // 目標類別為 index 2
    let loss = activated.cross_entropy(&targets);

    println!("Forward Pass 結束");
    println!("Loss: {:?}", loss.inner.borrow().data[0]);

    // 進行反向傳播 (Backward Pass)
    loss.backward();

    println!("Backward Pass 結束");
    println!("Weights Gradients:");
    println!("{:?}", weights.inner.borrow().grad);
}
*/