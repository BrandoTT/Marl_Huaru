## 制作新景象，将算法代码和仿真镜像放在一起
FROM xsim:v6.0

WORKDIR /home/Hok_Marl_ppo

RUN apt update install && apt-get install python3.8
echo 'PATH'
RUN pip install --no-cache-dir -r ./requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 SSH 服务
RUN apt-get update && apt-get install -y openssh-server

# 复制代码
COPY /home/ubuntu/Hok_Marl_ppo /home/Hok_Marl_ppo


