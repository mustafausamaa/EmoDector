import random

# import questions from questions.txt into a dictionary


def get_questions():
    questions = {}
    # emotion : question
    # each emotion has a 6 questions

    with open('Questions/questions.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                emotion, question = line.split(':')
                # append question to list of questions for that emotion
                if emotion in questions:
                    questions[emotion].append(question)
                else:
                    questions[emotion] = [question]

    return questions

# get a random question from the dictionary


def get_emotion_question(emotion):
    questions = get_questions()
    returned_question = []
    if emotion in questions:
        # select two random questions from the list of questions for that emotion
        returned_question = random.sample(questions[emotion], 2)
    return returned_question
