# 零件类别配置

检测、服务端模型切换和客户端可视化共用 `parts_config.json`。
类别数量不再写死；数组中有多少个启用项，就检测和抓取多少类。

## 增加普通零件

在 `parts` 数组追加：

```json
{
  "id": "cat5",
  "label": "an unambiguous English visual description",
  "mesh_file": "../assets/category_5.obj",
  "enabled": true,
  "area_range": [0.01, 0.10],
  "aspect_range": [0.4, 2.5],
  "color": "purple"
}
```

- `id`、`label` 必须唯一。
- `mesh_file` 相对配置文件所在目录，也可写绝对路径；启动时会检查文件是否存在。
- `area_range` 是框面积占整图面积的范围。
- `aspect_range`、`color` 可省略。
- 临时停用某类只需设置 `"enabled": false`；永久删除则移除对应对象。

## 可选检测设置

- `supplement: true`：joint 检测后再单独补检该类。
- `supplement_labels`：补检时使用的一组更具体描述；省略时使用 `label`。
- `keywords`：模型返回近似标签时用于匹配类别。
- `conflict_score`：不同类别框重叠时的保留优先级。
- `pose_mask_bbox_rerank`：对外观/几何高度对称的零件，可用 SAM
  紧框重排 FoundationPose 的高分候选。`top_k` 限制候选数，
  `mask_weight` 控制投影框一致性权重，`min_iou_gain` 防止小幅变化覆盖
  原始 scorer 结果。

`postprocess_role`、`merge_group`、`joint_min_area_from` 等字段是当前四类零件经过调优的专用规则。新增普通零件不应复制这些字段，除非它确实需要相同的外观消歧逻辑。

可通过环境变量使用另一份配置：

```bash
PARTS_CONFIG=/path/to/my_parts.json scripts/run_zerith_locate_anything.sh
```
