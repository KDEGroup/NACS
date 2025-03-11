import subprocess


def invokeTransformations(code):
    # 保存输入代码到临时文件
    input_file = "input_code.txt"
    with open(input_file, "w") as f:
        f.write(code)

    # 调用 Java 程序
    try:
        result = subprocess.run(
            ['java', '-jar', './ast_pretrain/untitled.jar', input_file],  # 调用编译后的 Java 程序
            check=True,  # 检查命令是否成功执行
            capture_output=True,  # 捕获标准输出
            text=True  # 输出作为文本而不是字节
        )
        # 获取增强后的代码
        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f"Error running Java code: {e}")
        return None
