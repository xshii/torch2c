# TODO

## 可视化修复
- [ ] Schedule 甘特图去重：过滤 tiled op 的 summary 条，避免与 per-tile 块重叠
- [ ] DMA 箭头 tiling 展开：为每个 tile 生成独立箭头，体现逐 tile 搬运

## 测试与验证
- [ ] 全量 ST 回归：确认 ST1~ST6 全部通过（特别是 ST3/ST5 spill 策略）
- [ ] 添加 `demo:st6` VSCode task，输出到 `output/st6/` 并打开可视化

## 功能扩展
- [ ] 完整 MHA 模型：加 V 投影 + concat + output projection，覆盖完整多头注意力
