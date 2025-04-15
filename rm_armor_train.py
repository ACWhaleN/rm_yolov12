from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-pose.yaml').load("yolov12n.pt")

# # Train the model
results = model.train(
  data='rm-armors-pose.yaml',
  batch=12, 
  imgsz=640,
  # 预训练模型
  pretrained=True,
  # 训练周期与学习率
  epochs=600, 
  cos_lr=True, # 余弦退火学习率调度器 
  lr0= 0.01,  # 初始学习率
  lrf= 0.1, # 最终学习率因子
  warmup_epochs=5,
  warmup_bias_lr=0.1, # 预热阶段偏置项的学习率
  # 数据增强
  augment= True,
  hsv_h= 0.0, # 还是不要动Hue了
  hsv_s= 0.2, # 小一点防止颜色弄混
  hsv_v= 0.6,  # 需要找到一个既能提高光照鲁棒性又不会混淆红蓝颜色的平衡点
  degrees=5.0, # 允许±5度旋转
  mosaic=1.0,
  shear=2.0, # 允许小幅度形变
  close_mosaic=100, #最后100 epochs关闭mosaic增强
  mixup=0.1,  # S:0.05; M:0.15; L:0.15; X:0.2
  fliplr=0.0,  # 关闭水平翻转（默认0.5）
  flipud=0.0,  # 关闭垂直翻转（默认0.0）
  perspective=0.0,  # 关闭透视变换（可能含翻转）
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  auto_augment='randaugment', 
  erasing= 0.2,
  # 损失函数权重 
  box= 12.0,
  cls= 3.0, # 增强分类损失的权重（ 你也不想痛击队友吧
  dfl= 3.0,
  pose= 64.0, # 增强关键点损失的权重（适当增强有利于解决关键点漂移
  kobj= 2.0, # 关键点对象性/可见性损失权重
  # 其他参数
  multi_scale=True, # 考虑到远近装甲板尺寸变化较大
  weight_decay=0.0005, # 这是标准的正则化方法
  dropout=0.1, # 数据集太小防止过拟合
  nbs= 64,
  translate=0.15,
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("data/images/test/205.jpg")
results[0].show()