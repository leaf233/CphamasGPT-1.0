# ============================================================
# 修复版：读取 NONMEM sdtab001 并绘制 GOF 图
# 作者：Assistant
# 说明：sdtab001 中有重复列名 DV，使用 check.names=TRUE 自动重命名
# ============================================================

# 清空环境，避免旧变量干扰
rm(list = ls())

# 加载所需包（如果未安装，请先 install.packages）
library(ggplot2)
library(patchwork)

# ========== 修改这里：你的模型输出目录 ==========
work_dir <- "D:/Programming/CphamasGPT/output/output_theo_opt5/1cmt"
# ===============================================

setwd(work_dir)

# ---- 1. 读取 sdtab001 ----
sdtab <- read.table(
  "sdtab001",
  header = TRUE,
  skip = 1,                 # 跳过第一行 "TABLE NO. X"
  fill = TRUE,
  check.names = TRUE,       # 关键：自动将重复列名变为 DV.1
  na.strings = c("", "NA", "NaN", ".")
)

cat("sdtab001 列名：\n")
print(colnames(sdtab))
cat("sdtab001 行数：", nrow(sdtab), "\n\n")

# ---- 2. 筛选观测行 ----
# 观测行特征：AMT == 0 且 DV 非缺失
# 注意：因为有两个 DV 列，第一个 DV 是观测浓度，第二个 DV.1 忽略
obs <- sdtab[!is.na(sdtab$AMT) & sdtab$AMT == 0 & !is.na(sdtab$DV), ]
cat("观测行数：", nrow(obs), "\n")

# 如果 AMT 列没有 0，可取消下一行注释并注释上一行
# obs <- sdtab[!is.na(sdtab$DV), ]

# ---- 3. 统一绘图主题 ----
gof_theme <- theme_bw(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

# ---- 4. 绘制四张 GOF 图 ----

# 4.1 DV vs PRED
p1 <- ggplot(obs, aes(x = PRED, y = DV)) +
  geom_point(alpha = 0.5, size = 1.8, color = "#2c7fb8") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "loess", se = TRUE, color = "darkgreen", 
              fill = "lightgreen", alpha = 0.3) +
  labs(x = "PRED", y = "DV", title = "DV vs PRED") +
  gof_theme

# 4.2 DV vs IPRED
p2 <- ggplot(obs, aes(x = IPRED, y = DV)) +
  geom_point(alpha = 0.5, size = 1.8, color = "#2c7fb8") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "loess", se = TRUE, color = "darkgreen", 
              fill = "lightgreen", alpha = 0.3) +
  labs(x = "IPRED", y = "DV", title = "DV vs IPRED") +
  gof_theme

# 4.3 CWRES vs TIME
p3 <- ggplot(obs, aes(x = TIME, y = CWRES)) +
  geom_point(alpha = 0.5, size = 1.8, color = "#2c7fb8") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_hline(yintercept = c(-2, 2), linetype = "dotted", color = "gray40") +
  geom_smooth(method = "loess", se = TRUE, color = "darkgreen", 
              fill = "lightgreen", alpha = 0.3) +
  labs(x = "TIME", y = "CWRES", title = "CWRES vs TIME") +
  gof_theme

# 4.4 CWRES vs PRED
p4 <- ggplot(obs, aes(x = PRED, y = CWRES)) +
  geom_point(alpha = 0.5, size = 1.8, color = "#2c7fb8") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_hline(yintercept = c(-2, 2), linetype = "dotted", color = "gray40") +
  geom_smooth(method = "loess", se = TRUE, color = "darkgreen", 
              fill = "lightgreen", alpha = 0.3) +
  labs(x = "PRED", y = "CWRES", title = "CWRES vs PRED") +
  gof_theme

# ---- 5. 组合并保存 ----
gof_plot <- (p1 + p2) / (p3 + p4)
print(gof_plot)

ggsave(
  filename = "GOF_theo_1cmt.png",
  plot = gof_plot,
  width = 10,
  height = 8,
  dpi = 300
)

cat("GOF 图已保存到：", file.path(work_dir, "GOF_theo_1cmt.png"), "\n")