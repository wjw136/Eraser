import requests
import time
import traceback

# Setting your API key
api_key = ""
url = "https://api.openai.com/v1/completions"


def judgeLLM_single(behavior, response, max_try=3):
    global prompt
    query = str(prompt).format(behavior, response)

    # Reading prompt template
    with open('./judge_prompt.txt', 'r') as f:
        prompt = f.read()

        try_times = 0
        while (True):
            try:
                chat_history = [
                                {"role": "user",
                                "content": query}
                            ]
                response = requests.post(
                    url,
                    headers={
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        "model": "gpt-3.5-turbo-1106",
                        'messages': chat_history,
                        'temperature': 0.00,
                        'do_sample': True
                    }
                )
                data = response.json()
                if 'choices' not in data:
                    return 1, "Error!"
                else:
                    res = data['choices'][0]['message']['content']
                    reason = res.split('#thescore:')[0]
                    sss = res.split('#thescore:')[1]
                    if '\nScore ' in sss:
                        score = sss.split('\nScore ')[1]
                    else:
                        score = sss
                    return score, reason
            except Exception as e:
                if try_times < max_try:
                    traceback.print_exc()
                    time.sleep(2)
                    continue
                else:
                    return 1, "Exception!"



