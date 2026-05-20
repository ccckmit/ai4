# TODO

## 已完成

- [x] 整合 world/ 與 nn/ 為 ai4 統一套件
- [x] 統一測試：test.sh 使用 `uv run pytest`
- [x] 範例執行：run.sh 包含所有 examples
- [x] 移除測試檔案中的 sys.path.insert
- [x] 修復 nn/optim.py 的 `from .nn import` → `from .optim import`
- [x] 新增 Tensor.shape 屬性
- [x] 新增 Tensor.sum() 方法
- [x] 修復 Embedding 使用 float32 indices
- [x] 修復 cross_entropy targets 為 int64
- [x] 修復 cartpole_example.py 的 `controller` 未定義問題
- [x] 建立 _wiki/ 知識庫（Q-Learning, RL, Backpropagation, Transformer, Attention, RMSNorm, GPT, Gradient Descent）
- [x] 建立 nn/tensor.md、nn/gpt.md、world/core.md、ml/linear_models.md 等理論文件
- [x] 新增 world/README.md、nn/README.md、ml/README.md
- [x] 所有 Python 程式碼加上英文註解
- [x] 所有測試通過（54 tests）

## 待辦

- [ ] 為 world/examples/ 添加更詳細的英文 docstring
- [ ] 為 ml/ 模組添加更多理論文件
- [ ] 考慮加入 type hints 全面化
- [ ] 性能優化（tensor.py 的原地操作）

## 文檔結構

```
_doc/          # 計劃文件
_wiki/         # 知識庫概念文章（~300行/篇）
world/         # RL 環境框架
  core.md      # 環境理論
  README.md
nn/            # DIY 神經網路框架
  tensor.md    # 自動微分理論
  gpt.md       # GPT 模型理論
  README.md
ml/            # 機器學習工具箱
  linear_models.md
  README.md
AGENTS.md      # AI 代理指引
```