import json
import pandas as pd

if __name__ == '__main__':
    with open('covid-chestxray-dataset/annotations/imageannotation_ai_lung_bounding_boxes.json', 'r',
              encoding='utf-8') as json_file:
        data = json.load(json_file)
        # df = pd.DataFrame(data)

        print(data)