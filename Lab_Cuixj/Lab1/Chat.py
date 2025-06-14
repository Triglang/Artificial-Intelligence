from openai import OpenAI

def chat(client, messages):
    """调用LLM并返回文本回复"""
    try:
        completion = client.chat.completions.create(
            model = 'DeepSeek-R1:latest',  # 确保与Ollama运行的模型名称一致
            temperature = 0.7,
            messages = messages
        )
        # 关键修复：使用 choices.message.content
        return completion.choices[0].message.content
    except Exception as e:
        return f"API调用失败: {str(e)}"

if __name__ == "__main__":
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    print("LLM对话程序（输入 exit 退出）")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'exit':
            print("对话结束")
            break

        messages.append({"role": "user", "content": user_input})
        response = chat(client, messages)
        messages.append({"role": "assistant", "content": response})
        print(f"\n助手:\n {response}")