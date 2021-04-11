import text2emotion as te
import pandas as pd

test_data = "novel_text_sent/cd_barnaby_rudge_a_tale_of_the_riots_of_eighty_by_charles_dickens_out.txt"
pred_data = "results/cd_barnaby_rudge_a_tale_of_the_riots_of_eighty_by_charles_dickens_pred.csv"

with open(test_data, "r", encoding="utf-8") as f:
    data = f.readlines()

result_list = []
for sent in data:
    text = sent.strip()
    pred = te.get_emotion(text)
    joy = pred["Happy"]
    anger = pred["Angry"]
    surprise = pred["Surprise"]
    sadness = pred["Sad"]
    fear = pred["Fear"]
    emotions_dict = {
        "text": text,
        "joy": joy,
        "anger": anger,
        "surprise": surprise,
        "sadness": sadness,
        "fear": fear
    }
    result_list.append(emotions_dict)

output = pd.DataFrame(result_list)
output.to_csv(pred_data)
