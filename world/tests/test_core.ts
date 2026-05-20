import { Tensor } from '../../nn/tensor';

const t1 = Tensor.from([[1, 2], [3, 4]], true);
const t2 = Tensor.from([[5, 6], [7, 8]], true);
const sum = t1.add(t2);
console.log('add:', sum.data);
sum.backward();
console.log('grad t1:', t1.grad);

const t3 = Tensor.from([[1], [2]]);
const t4 = Tensor.from([[3, 4]]);
const mm = t3.matmul(t4);
console.log('matmul:', mm.data);

const t5 = Tensor.from([[1, 2, 3]]);
const r = t5.relu();
console.log('relu:', r.data);

console.log('world/core tests passed');