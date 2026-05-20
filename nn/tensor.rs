//! nn/tensor.rs
//! Tensor with automatic differentiation (autograd).
//! Records operation history to support backpropagation through a computation graph.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;
use rand::SeedableRng;

pub struct Tensor {
    pub data: Rc<RefCell<Vec<f32>>>,
    pub grad: Rc<RefCell<Vec<f32>>>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub _backward: Option<Box<dyn Fn()>>,
    pub _prev: HashSet<usize>,
    pub id: usize,
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
            grad: self.grad.clone(),
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
            _backward: None,
            _prev: self._prev.clone(),
            id: self.id,
        }
    }
}

static mut TENSOR_ID: usize = 0;

fn next_id() -> usize {
    unsafe {
        TENSOR_ID += 1;
        TENSOR_ID
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let grad = vec![0.0; data.len()];
        let id = next_id();
        
        Tensor {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(grad)),
            shape,
            requires_grad,
            _backward: None,
            _prev: HashSet::new(),
            id,
        }
    }

    pub fn from_vec(data: Vec<f32>, requires_grad: bool) -> Self {
        let shape = vec![data.len()];
        Self::new(data, shape, requires_grad)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product::<usize>();
        Self::new(vec![0.0; size], shape.to_vec(), false)
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product::<usize>();
        Self::new(vec![1.0; size], shape.to_vec(), false)
    }

    pub fn randn(shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::from_entropy();
        let size = shape.iter().product::<usize>();
        let data: Vec<f32> = (0..size).map(|_| rng.sample(rand_distr::Normal::new(0.0, 1.0).unwrap()) as f32).collect();
        Self::new(data, shape.to_vec(), false)
    }

    pub fn zero_grad(&mut self) {
        let mut g = self.grad.borrow_mut();
        for v in g.iter_mut() {
            *v = 0.0;
        }
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        
        fn build_topo(v: &Tensor, visited: &mut HashSet<usize>, topo: &mut Vec<usize>) {
            if !visited.contains(&v.id) {
                visited.insert(v.id);
                for &prev_id in &v._prev {
                    // We need to find the tensor by ID - this is a simplification
                }
                topo.push(v.id);
            }
        }
        
        build_topo(self, &mut visited, &mut topo);
        
        // Initialize gradient to ones
        let mut g = self.grad.borrow_mut();
        for v in g.iter_mut() {
            *v = 1.0;
        }
        
        // Execute backward in reverse topological order
        for id in topo.iter().rev() {
            // In a full implementation, we'd look up tensors by ID and call their _backward
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let data: Vec<f32> = self.data.borrow().iter().zip(other.data.borrow().iter())
            .map(|(a, b)| a + b)
            .collect();
        
        let mut shape = self.shape.clone();
        if other.shape.len() > shape.len() {
            shape = other.shape.clone();
        }
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut out = Tensor::new(data, shape, requires_grad);
        out._prev.insert(self.id);
        out._prev.insert(other.id);
        
        if requires_grad {
            let self_data = self.data.clone();
            let other_data = other.data.clone();
            let self_grad = self.grad.clone();
            let other_grad = other.grad.clone();
            let out_grad = out.grad.clone();
            
            out._backward = Some(Box::new(move || {
                let og = out_grad.borrow();
                let mut sg = self_grad.borrow_mut();
                let mut og = other_grad.borrow_mut();
                for i in 0..sg.len() {
                    sg[i] += og[i];
                    og[i] += og[i];
                }
            }));
        }
        
        out
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let data: Vec<f32> = self.data.borrow().iter().zip(other.data.borrow().iter())
            .map(|(a, b)| a * b)
            .collect();
        
        let shape = if self.shape.len() > other.shape.len() { self.shape.clone() } else { other.shape.clone() };
        let requires_grad = self.requires_grad || other.requires_grad;
        
        let mut out = Tensor::new(data, shape, requires_grad);
        out._prev.insert(self.id);
        out._prev.insert(other.id);
        
        if requires_grad {
            let self_data = self.data.clone();
            let other_data = other.data.clone();
            let self_grad = self.grad.clone();
            let other_grad = other.grad.clone();
            let out_grad = out.grad.clone();
            let shape1 = self.shape.clone();
            let shape2 = other.shape.clone();
            
            out._backward = Some(Box::new(move || {
                let od = out_grad.borrow();
                let sd = self_data.borrow();
                let od2 = other_data.borrow();
                let mut sg = self_grad.borrow_mut();
                let mut og = other_grad.borrow_mut();
                
                for i in 0..sg.len() {
                    sg[i] += od[i] * od2[i % od2.len()];
                    og[i] += od[i] * sd[i % sd.len()];
                }
            }));
        }
        
        out
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let b1 = self.shape.len();
        let m = *self.shape.last().unwrap();
        let k = *other.shape.last().unwrap();
        
        let a = self.data.borrow();
        let b = other.data.borrow();
        let mut result = vec![0.0; m * k];
        
        for i in 0..m {
            for j in 0..k {
                for l in 0..b1 {
                    result[i * k + j] += a[i * b1 + l] * b[l * k + j];
                }
            }
        }
        
        let shape = vec![m, k];
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut out = Tensor::new(result, shape, requires_grad);
        out._prev.insert(self.id);
        out._prev.insert(other.id);
        
        out
    }

    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self.data.borrow().iter().map(|&x| x.max(0.0)).collect();
        
        let mut out = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.requires_grad {
            out._prev.insert(self.id);
            
            let self_data = self.data.clone();
            let self_grad = self.grad.clone();
            let out_grad = out.grad.clone();
            
            out._backward = Some(Box::new(move || {
                let sd = self_data.borrow();
                let og = out_grad.borrow();
                let mut sg = self_grad.borrow_mut();
                
                for i in 0..sg.len() {
                    if sd[i] > 0.0 {
                        sg[i] += og[i];
                    }
                }
            }));
        }
        
        out
    }

    pub fn sum(&self) -> Tensor {
        let sum: f32 = self.data.borrow().iter().sum();
        
        let mut out = Tensor::new(vec![sum], vec![1], self.requires_grad);
        if self.requires_grad {
            out._prev.insert(self.id);
            
            let self_grad = self.grad.clone();
            let out_grad = out.grad.clone();
            let size = self.data.borrow().len();
            
            out._backward = Some(Box::new(move || {
                let og = out_grad.borrow();
                let mut sg = self_grad.borrow_mut();
                let val = og[0];
                for s in sg.iter_mut() {
                    *s += val;
                }
            }));
        }
        
        out
    }

    pub fn transpose(&self) -> Tensor {
        let shape = vec![self.shape[1], self.shape[0]];
        let data: Vec<f32> = self.data.borrow().chunks(self.shape[0])
            .flat_map(|c| c.iter().copied())
            .collect();
        
        let mut out = Tensor::new(data, shape, self.requires_grad);
        if self.requires_grad {
            out._prev.insert(self.id);
        }
        
        out
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        let data = self.data.borrow().clone();
        
        let mut out = Tensor::new(data, shape, self.requires_grad);
        if self.requires_grad {
            out._prev.insert(self.id);
        }
        
        out
    }

    pub fn softmax(&self, axis: usize) -> Tensor {
        let data = self.data.borrow();
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let exps: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();
        
        let mut out = Tensor::new(probs, self.shape.clone(), self.requires_grad);
        if self.requires_grad {
            out._prev.insert(self.id);
            
            // Simplified softmax backward
            let self_grad = self.grad.clone();
            let out_grad = out.grad.clone();
            
            out._backward = Some(Box::new(move || {
                let og = out_grad.borrow();
                let mut sg = self_grad.borrow_mut();
                for i in 0..sg.len() {
                    sg[i] += og[i];
                }
            }));
        }
        
        out
    }

    pub fn cross_entropy(&self, targets: &[usize]) -> Tensor {
        let data = self.data.borrow();
        
        if targets.is_empty() || data.is_empty() {
            return Tensor::new(vec![0.0], vec![1], self.requires_grad);
        }
        
        let vocab_size = if self.shape.len() == 2 {
            self.shape[1]
        } else if self.shape.len() == 3 {
            self.shape[2]
        } else {
            data.len() / targets.len().max(1)
        };
        
        if vocab_size == 0 {
            return Tensor::new(vec![0.0], vec![1], self.requires_grad);
        }
        
        let n = (data.len() / vocab_size).max(1);
        
        let max_logits: Vec<f32> = (0..n)
            .map(|i| {
                let start = i * vocab_size;
                let end = start + vocab_size;
                data[start..end.min(data.len())].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            })
            .collect();
        
        let mut loss = 0.0;
        for i in 0..n.min(targets.len()) {
            let offset = i * vocab_size;
            let end = (offset + vocab_size).min(data.len());
            let exp_sum: f32 = data[offset..end].iter()
                .map(|&x| (x - max_logits[i]).exp())
                .sum();
            
            let target = targets[i];
            if offset + target < data.len() {
                let prob = (data[offset + target] - max_logits[i]).exp() / exp_sum.max(1e-10);
                loss -= prob.max(1e-10).ln();
            }
        }
        let m = n.min(targets.len()).max(1);
        if m > 0 {
            loss /= m as f32;
        }
        
        let mut out = Tensor::new(vec![loss], vec![1], self.requires_grad);
        if self.requires_grad {
            out._prev.insert(self.id);
        }
        
        out
    }

    pub fn pow(&self, power: f32) -> Tensor {
        let data: Vec<f32> = self.data.borrow().iter().map(|&x| x.powf(power)).collect();
        
        let mut out = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.requires_grad {
            out._prev.insert(self.id);
            
            let self_data = self.data.clone();
            let self_grad = self.grad.clone();
            let out_grad = out.grad.clone();
            let p = power;
            
            out._backward = Some(Box::new(move || {
                let sd = self_data.borrow();
                let og = out_grad.borrow();
                let mut sg = self_grad.borrow_mut();
                
                for i in 0..sg.len() {
                    sg[i] += og[i] * p * sd[i].powf(p - 1.0);
                }
            }));
        }
        
        out
    }

    pub fn neg(&self) -> Tensor {
        let data: Vec<f32> = self.data.borrow().iter().map(|&x| -x).collect();
        
        let mut out = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.requires_grad {
            out._prev.insert(self.id);
            
            let self_grad = self.grad.clone();
            let out_grad = out.grad.clone();
            
            out._backward = Some(Box::new(move || {
                let og = out_grad.borrow();
                let mut sg = self_grad.borrow_mut();
                for i in 0..sg.len() {
                    sg[i] -= og[i];
                }
            }));
        }
        
        out
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, requires_grad={})", self.shape, self.requires_grad)
    }
}

pub fn cat(tensors: &[Tensor], axis: usize) -> Tensor {
    let mut all_data = Vec::new();
    let mut shapes = Vec::new();
    let mut requires_grad = false;
    
    for t in tensors {
        all_data.extend(t.data.borrow().clone());
        shapes.push(t.shape.clone());
        requires_grad = requires_grad || t.requires_grad;
    }
    
    let new_size = tensors.iter().map(|t| t.shape[axis]).sum();
    let mut new_shape = shapes[0].clone();
    new_shape[axis] = new_size;
    
    let mut out = Tensor::new(all_data, new_shape, requires_grad);
    
    for t in tensors {
        out._prev.insert(t.id);
    }
    
    out
}