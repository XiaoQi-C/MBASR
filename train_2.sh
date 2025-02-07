#!/bin/bash

tag=(5)
k_values=(0.2 0.4 0.6 0.8)
#k_values=(0.8)
p_values=(0 0.2 0.4 0.6 0.8 1)

# 设置最大并行数
max_parallel=2

# 定义函数来运行训练脚本并重试
run_training() {
  tag=$1
  k_value=$2
  p_value=$3

  params="--tag $tag --k $k_value --p $p_value"
  log_file="Res/sim_unify/SASRec_BAR/Tmall/tag_${tag}_k_${k_value}_p_${p_value}.txt"

  # 尝试运行训练脚本，并将输出重定向到日志文件
  echo "开始训练: tag=${tag}, k=${k_value}, p=${p_value}"
  export CUDA_VISIBLE_DEVICES=3
  python SASRec_BAR.py $params  > $log_file

  if [ $? -eq 0 ]; then
    echo "训练成功: tag=${tag}, k=${k_value}, p=${p_value}"
  else
    echo "训练失败: tag=${tag}, k=${k_value}, p=${p_value}"
    if [ "$interrupted" = true ]; then
      echo "已中断，不再重试"
    else
      echo "重新运行训练"
      interrupted=true
      run_training $tag $k_value $p_value
    fi
  fi
}

# 初始化并行进程计数器和进程总数
parallel_count=0
total_processes=$(( ${#tag[@]} * ${#k_values[@]} * ${#p_values[@]} ))

# 设置中断标志
interrupted=false

# 定义中断处理函数
on_interrupt() {
  echo "接收到中断信号，正在退出..."
  interrupted=true
  exit 1
}

# 捕获中断信号并调用中断处理函数
trap on_interrupt INT

# 遍历参数组合并并行运行
for tag in "${tag[@]}"; do
  for k_value in "${k_values[@]}"; do
    for p_value in "${p_values[@]}"; do
      if [ "$parallel_count" -eq "$max_parallel" ]; then
        wait -n
        parallel_count=$((parallel_count-1))
      fi

      # 并行运行训练
      run_training "$tag" "$k_value" "$p_value" &
      parallel_count=$((parallel_count+1))

      # 显示进度信息
      progress=$(( parallel_count * 100 / total_processes ))
      echo "当前进度: $progress%"
    done
  done
done

# 等待剩余的后台进程完成
wait
echo "运行完成"
