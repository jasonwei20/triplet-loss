import json
from random import shuffle


#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

def clean_dataset(file_path, output_path_train, output_path_test):
	
	lines = open(file_path, 'r').readlines()
	category_to_headlines = {}

	for line in lines:
		d = json.loads(line[:-1])
		category = d['category']
		headline = d['headline']
		if len(headline) > 10:
			if category in category_to_headlines:
				category_to_headlines[category].append(headline)
			else:
				category_to_headlines[category] = [headline]

	category_to_id = {category: i for i, category in enumerate(list(sorted(list(category_to_headlines.keys()))))}

	train_writer = open(output_path_train, 'w')
	test_writer = open(output_path_test, 'w')

	for category, headlines in category_to_headlines.items():

		_id = category_to_id[category]

		shuffle(headlines)
		test_headlines = headlines[:300]
		train_headlines = headlines[300:1000]

		for train_headline in train_headlines:
			train_writer.write('\t'.join([str(_id), get_only_chars(train_headline)]) + '\n')

		for test_headline in test_headlines:
			test_writer.write('\t'.join([str(_id), get_only_chars(test_headline)]) + '\n')


if __name__ == "__main__":
	clean_dataset('News_Category_dataset_v2.json', 'huffpost/train.txt', 'huffpost/test.txt')