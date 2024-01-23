

import time
from openai import OpenAI

client = OpenAI()

file = client.files.create(file=open("test.csv", "rb"), purpose="assistants")

print(file)


time.sleep(10)


def save_bill(totalCost, totalIncome):
    print(totalCost, totalIncome)
    return 'success'


function = {
    "type": "function",
    "function": {
            "name": "save_bill",
            "description": "保存总成本和总的收入",
            "parameters": {
                "type": "object",
                "properties": {
                    "totalCost": {
                        "type": "number",
                        "description": "总成本",
                    },
                    "totalIncome": {
                        "type": "number",
                        "description": "总收入",
                    }
                },
                "required": ["totalCost", "totalIncome"],
            },
    }
}
available_functions = {"save_bill": save_bill}


assistant = client.beta.assistants.create(
    name="花店财务助手",
    description="按照每种花的售出量，统计成本和收入，计算出总利润",
    model="gpt-4-1106-preview",
    tools=[{"type": "code_interpreter"}, {"type": "retrieval"}, function],
    file_ids=[file.id]
)

thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "我卖出去了红玫瑰3支、郁金香2支、百合6支，计算下总成本和总收入，给出具体的计算过程"
        }
    ]
)


run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)


print('run ======', run)


time.sleep(10)

if run.status == 'requires_action':
    tool_outputs = []

    for call in run.required_action.submit_tool_outputs.tool_calls:
        print('call ======', call)
        if call.type != "function":
            continue
        # 获取真实函数
        function = available_functions[call.function.name]
        output = {
            "tool_call_id": call.id,
            "output": function(**call.function.arguments),
        }
        tool_outputs.append(output)
    # 将函数调用的结果回传给Assistant
    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=tool_outputs
    )


time.sleep(10)


# 获取run的最新状态。
run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
)

if run.status == 'completed':
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    print(messages)


# 创建新的消息

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="另外还有2支向日葵，补充下这份账单"
)
# 创建run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

time.sleep(10)

# 获取执行结果
run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
)


if run.status == 'completed':
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    print(messages)
