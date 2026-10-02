import requests


BASE_URL = "https://agents-course-unit4-scoring.hf.space"


def get_random_question():
    url = f"{BASE_URL}/random-question"
    response = requests.get(url, timeout=15)

    if response.status_code == 200:
        question_data = response.json()
        print("Random question received!")
        return question_data
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def get_all_questions():
    url = f"{BASE_URL}/questions"
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        questions_list = response.json()
        print(f"\nSuccess! Obtained questions: {len(questions_list)}")
        return questions_list
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None